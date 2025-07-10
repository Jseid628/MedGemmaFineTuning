import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # optional 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# For set up
from datasets import load_dataset
from typing import Any

# For Loading Model
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# For fine tuning
from peft import LoraConfig
from trl import SFTTrainer

# For setting training parameters
from trl import SFTConfig

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch sees this as device:", torch.cuda.current_device())
print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# ---------------------- Set Up ---------------------- #

train_size = 9000 
validation_size = 1000 

# Downloaded and organized a subset of the patchcamelyon data set (first 10K images) into
# a folder called patchcamelyon_subset. Has sub folders "normal" and "tumor"
data = load_dataset("./patchcamelyon_subset", split="train")
data = data.train_test_split(
    train_size=train_size,
    test_size=validation_size,
    shuffle=True,
    seed=42,
)
# rename the 'test' set to 'validation'
data["validation"] = data.pop("test")

#Smaller to test pipeline
data["train"] = data["train"].select(range(1000))
data["validation"] = data["validation"].select(range(200))

# ------------ Optional: display dataset details ------------ #
print(data) 
# This is actually a dictionary - it contains {'image':blah, 'label':hmmm}
print(f"data['train'][0]: {data['train'][0]}")
# First image in the training data
image = data['train'][0]['image']
# First label in the training data
label = data['train'][0]['label']
image.save("sample_image.png")
print("Image saved to sample_image.png")
print(label)
print(data['train'].features['label'])
# ----------------------------------------------------------- #

HISTOPATHOLOGY_CLASSES = [
    # One option for each class
    "A: no tumor present",
    "B: tumor present"
]

options = "\n".join(HISTOPATHOLOGY_CLASSES)
PROMPT = f"Is a tumor present in this histopathology image?\n{options}"

# 'example' is the name of the input here - input is a dict.
# The key for this dict is a str and the value can be of Any type
def format_data(example: dict[str, Any]) -> dict[str, Any]:
    # adds a new entry to the dict
    example["messages"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": PROMPT,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    # label of 0 will map to: (A: no tumor present), label of 1 will map to: (B: tumor present)
                    "text": HISTOPATHOLOGY_CLASSES[example["label"]],
                },
            ],
        },
    ]
    # Returns a dict with the same structure - but now {'image':blah, 'label':hmmm, 'message':blumph}
    return example

data = data.map(format_data)
print(data['train'][0])

# ---------------------- Loading Model ---------------------- #

model_id = "google/medgemma-4b-it"

# Check if GPU supports bfloat16
if torch.cuda.get_device_capability()[0] < 8:
    raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")
else: 
    print('GPU supports bfloat 16. You are good to go :)')

# A dictionary of model arguments - ie, 'attn_implementation' maps to 'eager'
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map={"": torch.device("cuda:0")},
)

# Add a dictionary entry 'quantization_config' - sets the values of 5 parameters in BitsAndBytesConfig() 
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
    bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
)

model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)

# This is where .apply_chat_template looks back to
processor = AutoProcessor.from_pretrained(model_id)

# Use right padding to avoid issues during training
processor.tokenizer.padding_side = "right"

# ----------------------  Set Up for Fine Tuning ---------------------- #

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)

# Step 1. Clone input_ids and assign to labels.
# Step 2. Mask unnecessary info
# Step 3. Add the now redacted info as a new entry in the batch called 'labels'

def collate_fn(examples: list[dict[str, Any]]):
    texts = []
    images = []
    for example in examples:
        images.append(example["image"].convert("RGB"))
        # Applies the chat template from messages and appends that to texts. 
        # Texts is a list of prompts with both A / B options, and the correct choice A or B.
        texts.append(processor.apply_chat_template(
            example["messages"],
            add_generation_prompt=False,
            tokenize=False
        ).strip())

    # Tokenize the texts and process the images
    # Contains 'input_ids'
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # These labels are tokenized version of the input
    labels = batch["input_ids"].clone()

    # Masking step begins here
    special_tokens = processor.tokenizer.special_tokens_map

    boi_token = special_tokens['boi_token']
    eoi_token = special_tokens['eoi_token']

    boi_token_id, eoi_token_id = processor.tokenizer.convert_tokens_to_ids([boi_token, eoi_token])

    # We don't want to predict image values. Any info with image token is masked since part of image.
    # Also masking padding tokens / other special tokens.
    ignore_token_ids = {
        processor.tokenizer.pad_token_id,
        boi_token_id,
        eoi_token_id,
        262144
    }

    # **Tensor masking** operation for tokens not used in the loss computation.
    for token_id in ignore_token_ids:
        labels[labels == token_id] = -100

    # 'labels' now contains how we want the model to behave: 'user: Heres an image - is it A or B?'  'model: it is A' All other info masked in labels section of batch.
    batch["labels"] = labels

    return batch

num_train_epochs = 1  # @param {type: "number"}
learning_rate = 2e-4  # @param {type: "number"}

args = SFTConfig(
    output_dir="medgemma-4b-it-sft-lora-PatchCamelyon",            # Directory and Hub repository id to save the model to
    num_train_epochs=num_train_epochs,                       # Number of training epochs
    per_device_train_batch_size=4,                           # Batch size per device during training
    per_device_eval_batch_size=4,                            # Batch size per device during evaluation
    gradient_accumulation_steps=4,                           # Number of steps before performing a backward/update pass
    gradient_checkpointing=True,                             # Enable gradient checkpointing to reduce memory usage
    optim="adamw_torch_fused",                               # Use fused AdamW optimizer for better performance
    logging_steps=50,                                        # Number of steps between logs
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

print("Batch test:", next(iter(trainer.get_train_dataloader())))