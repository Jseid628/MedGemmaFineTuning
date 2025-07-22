from typing import Any
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# --------------- patchcamelyon formatting function --------------- #

HISTOPATHOLOGY_CLASSES = [
    # One option for each class
    "A: no tumor present",
    "B: tumor present"
]

options = "\n".join(HISTOPATHOLOGY_CLASSES)
PROMPT = f"Is a tumor present in this histopathology image?\n{options}"

# 'example' is the name of the input here - input is a dict.
# The key for this dict is a str and the value can be of Any type
def format_data_patchcamelyon(example: dict[str, Any]) -> dict[str, Any]:
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

# --------------- diabetic retinopathy formatting function --------------- #

DR_CLASSES = [
    "A: No diabetic retinopathy present",
    "B: Diabetic retinopathy present"
]

options = "\n".join(DR_CLASSES)
PROMPT = f"Is diabetic retinopathy present in this image?\n{options}"

def format_data_exeye(example: dict[str, Any]) -> dict[str, Any]:
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
                    # if example['label'] = 0, then will map to "A: no diabetic retinopathy present"
                    "text": DR_CLASSES[example['label']], 
                },
            ],
        },
    ]
    # Returns a dict with the same structure - but now {'image':blah, 'label':hmmm, 'message':blumph}
    return example
# ---------------- model loading function ---------------- #

def load_model_and_processor(model_id = "google/medgemma-4b-it"):
    # Check if GPU supports bfloat16
    # major must be 8 to support bfloat16
    if torch.cuda.get_device_capability()[0] < 8:
        raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")
    else: 
        print('Loading Model and Processor... GPU supports bfloat 16. You are good to go :)')

    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        # optimal device map when using one GPU.
        device_map='balanced',
    )

    # Add a dictionary entry 'quantization_config' - sets the values of 5 parameters in BitsAndBytesConfig() 
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
        bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
        llm_int8_enable_fp32_cpu_offload=True, 
    )

    # model is assigned the pretrained model (google/medgemma-4b-it) with the specifications (model_kwargs)
    # ** unpacks the dictionary values as arguments to the from_pretrained function
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)

    # This is where .apply_chat_template looks back to
    processor = AutoProcessor.from_pretrained(model_id)

    # Use right padding to avoid issues during training
    processor.tokenizer.padding_side = "right"

    return model, processor
