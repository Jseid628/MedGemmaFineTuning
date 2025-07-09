
# For set up
from datasets import load_dataset
from typing import Any

# For Loading Model
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# For fine tuning
from peft import LoraConfig

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
    device_map="auto",
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
   

def collate_fn(examples: list[dict[str, Any]]):
    texts = []
    images = []
    for example in examples:
        images.append([example["image"].convert("RGB")])
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

    # The labels are the input_ids, with the padding and image tokens masked in
    # the loss computation
    labels = batch["input_ids"].clone()