import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# For set up
from datasets import load_dataset
from typing import Any

# For Loading Model
import torch
torch.cuda.empty_cache()
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# For fine tuning
from peft import LoraConfig
from trl import SFTTrainer

# For setting training parameters
from trl import SFTConfig

# from utils
from utils import format_data
from utils import load_model_and_processor

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True, help='Path to training data')
parser.add_argument('--output_dir', required=True, help='Output directory for model')
parser.add_argument('--final_model_dir', default='base_model', help='Base model name')
args = parser.parse_args()

# ---------------------- Set Up ---------------------- #

train_size = 9000 
validation_size = 1000 

# Downloaded and organized a subset of the patchcamelyon data set (first 10K images) into
# a folder called patchcamelyon_subset. Has sub folders "normal" and "tumor"
data = load_dataset(args.data_path, split="train")
data = data.train_test_split(
    train_size=train_size,
    test_size=validation_size,
    shuffle=True,
    seed=42,
)
# rename the 'test' set to 'validation'
data["validation"] = data.pop("test")

#Smaller subset to test pipeline if desired
# data["train"] = data["train"].select(range(1000))
# data["validation"] = data["validation"].select(range(200))

# So the appropriate formatting function is used
def get_formatter(data_path):
    if 'patchcamelyon' in data_path.lower():
        return format_data_patchcamelyon
    if 'externaleye' in data_path.lower():
        return format_data_exeye
    else:
        raise ValueError(f"No formatter found for dataset: {data_path}")

formatter = get_formatter(args.data_path)
data = data.map(formatter)

# Load model from utils.py
model, processor = load_model_and_processor()

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)

# ----------------------  data collating function ---------------------- #

# Step 1. Clone input_ids and assign to labels.
# Step 2. Mask unnecessary info
# Step 3. Add the now redacted info as a new entry in the batch called 'labels'

def collate_fn(examples: list[dict[str, Any]]):
    
    input_ids_list = []
    attention_mask_list = []
    pixel_values_list = []
    token_type_ids_list = []
    
    for example in examples:
        image = example["image"].convert("RGB")
        # Applies the chat template from messages and appends that to texts. 
        # Texts is a list of prompts with both A / B options, and the correct choice A or B.
        text = processor.apply_chat_template(
            example["messages"],
            add_generation_prompt=False,
            tokenize=False
        ).strip()

        processed = processor(text=text, images=image, return_tensors="pt", padding=True)

        # Add single processed example lists
        input_ids_list.append(processed["input_ids"][0])
        attention_mask_list.append(processed["attention_mask"][0])
        token_type_ids_list.append(processed['token_type_ids'][0])
        pixel_values_list.append(processed["pixel_values"][0])

    # Pad sequences - after having added all examples to the lists. Ensures all examples have same length values for given keys.
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids_list, batch_first=True, padding_value=0)
    pixel_values = torch.stack(pixel_values_list)

    # Label / Masking Step 

    # We want to predict the text output part of the input. We will later mask the image part.
    labels = input_ids.clone()

    # Mask special tokens
    special_tokens = processor.tokenizer.special_tokens_map
    boi_token_id, eoi_token_id = processor.tokenizer.convert_tokens_to_ids([
        special_tokens['boi_token'], special_tokens['eoi_token']
    ])

    # We don't want to predict image values. Any info with image token is masked since part of image.
    # Also masking padding tokens / other special tokens.
    ignore_token_ids = {
        processor.tokenizer.pad_token_id,
        boi_token_id,
        eoi_token_id,
        262144,  # Optional: image token
    }

    # **Tensor masking** operation for tokens not used in the loss computation.
    # 'labels' now contains how we want the model to behave: 'user: Heres an image - is it A or B?'  'model: it is A' All other info masked in labels section of batch.
    for token_id in ignore_token_ids:
        labels[labels == token_id] = -100


    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "pixel_values": pixel_values,
        "labels": labels,
    }

# ----------------------  parameters ---------------------- #

num_train_epochs = 4  # @param {type: "number"}
learning_rate = 2e-4  # @param {type: "number"}

args = SFTConfig(
    
    output_dir=args.output_dir,            # Directory and Hub repository id to save the model to
    num_train_epochs=num_train_epochs,                       # Number of training epochs
    per_device_train_batch_size=4,                           # Batch size per device during training
    per_device_eval_batch_size=4,                            # Batch size per device during evaluation
    gradient_accumulation_steps=4,                           # Number of steps before performing a backward/update pass
    gradient_checkpointing=True,                             # Enable gradient checkpointing to reduce memory usage
    optim="adamw_torch_fused",                               # Use fused AdamW optimizer for better performance
    logging_steps=10,                                        # Number of steps between logs
    save_strategy="epoch",                                   # Save checkpoint every epoch
    eval_strategy="steps",                                   # Evaluate every `eval_steps`
    eval_steps=50,                                           # Number of steps between evaluations
    learning_rate=learning_rate,                             # Learning rate based on QLoRA paper
    bf16=True,                                               # Use bfloat16 precision
    max_grad_norm=0.3,                                       # Max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                                       # Warmup ratio based on QLoRA paper
    lr_scheduler_type="linear",                              # Use linear learning rate scheduler
    push_to_hub=False,                                        # Push model to Hub
    report_to="tensorboard",                                 # Report metrics to tensorboard
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Set gradient checkpointing to non-reentrant to avoid issues
    dataset_kwargs={"skip_prepare_dataset": True},           # Skip default dataset preparation to preprocess manually
    remove_unused_columns = False,                           # Columns are unused for training but needed for data collator
    label_names=["labels"],                                  # Input keys that correspond to the labels
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=data["train"],
    eval_dataset=data["validation"].shuffle().select(range(200)),  # Use subset of validation set for faster run
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
)

trainer.train()

# ---- Save the merged final model ----

# Merge the LoRA weights into the base model
model = model.merge_and_unload()

# Update the trainer with the merged model
trainer.model = model

# Save the final merged model
trainer.save_model(args.final_model_dir)