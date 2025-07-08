
from datasets import load_dataset
from typing import Any

train_size = 9000  # @param {type: "number"}
validation_size = 1000  # @param {type: "number"}

# Downloaded and organized a subset of the patchcamelyon data set (first 10K images) into
# a folder called patchcamelyon_subset. Has sub folders "normal" and "tumor"
data = load_dataset("./patchcamelyon_subset", split="train")
data = data.train_test_split(
    train_size=train_size,
    test_size=validation_size,
    shuffle=True,
    seed=42,
)
# Use the test split as the validation set
data["validation"] = data.pop("test")

# ------------ Optional: display dataset details ------------ #
print(data) 
# This is actually a dictionary - it contains {'image':blah, 'label':hmmm}
print(f"data['train'][0]: {data['train'][0]}")
image = data['train'][0]['image']
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