
# For set up
from datasets import load_dataset
from typing import Any

# Other
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

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

# A dictionary of model arguments
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


# Add a dictionary entry 
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
    bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
)