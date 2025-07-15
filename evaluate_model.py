# Imports
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import utils
from datasets import load_dataset
from datasets import ClassLabel
from transformers import AutoModelForImageTextToText, AutoProcessor, pipeline
# from transformers import PaliGemmaForConditionalGeneration - if loading from checkpoint
from peft import PeftModel
import evaluate
from typing import Any
from tqdm import tqdm

# Constants
HISTOPATHOLOGY_CLASSES = [
    # One option for each class
    "A: no tumor present",
    "B: tumor present"
]

options = "\n".join(HISTOPATHOLOGY_CLASSES)
PROMPT = f"Is a tumor present in this histopathology image?\n{options}"

# Top level functions
def format_test_data(example: dict[str, Any]) -> dict[str, Any]:
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
    ]
    return example

def main():
    # Renaming for my own sanity
    raw = load_dataset("./patchcamelyon_test")
    test_data = raw["train"]
    test_data = test_data.shuffle(seed=42).select(range(1000))

    # Ground truth labels
    ground_truth_labels = test_data["label"]

    # format test_data
    test_data = test_data.map(format_test_data)

    # ----------- For Post Processing ---------- #

    # Rename the class names to the tissue classes, `X: tissue type`
    test_data = test_data.cast_column(
        "label",
        ClassLabel(names=HISTOPATHOLOGY_CLASSES)
    )
    LABEL_FEATURE = test_data.features["label"]

    # Mapping to alternative label format, `(X) tissue type`
    ALT_LABELS = dict([
        (label, f"({label.replace(': ', ') ')}") for label in HISTOPATHOLOGY_CLASSES
    ])

    def postprocess(prediction: list[dict[str, str]], do_full_match: bool=False) -> int:
        response_text = prediction[0]["generated_text"]
        if do_full_match:
            return LABEL_FEATURE.str2int(response_text)
        for label in HISTOPATHOLOGY_CLASSES:
            # Search for `X: tissue type` or `(X) tissue type` in the response
            if label in response_text or ALT_LABELS[label] in response_text:
                return LABEL_FEATURE.str2int(label)
        return -1

    # ---------------- Loading Saved Model -------------- # 

    model = AutoModelForImageTextToText.from_pretrained("medgemma-4b-it-sft-lora-PatchCamelyon-final")
    processor = AutoProcessor.from_pretrained("medgemma-4b-it-sft-lora-PatchCamelyon-final")
    model.eval()

    # ----------- Loading Model from Checkpoint ----------- #

    # base_model, processor = utils.load_model_and_processor()
    # lora_check_point_path = './medgemma-4b-it-sft-lora-PatchCamelyon/checkpoint-252'

    # model = PeftModel.from_pretrained(base_model, lora_check_point_path)
    # model = model.merge_and_unload()  # Applies the LoRA weights to the original model
    # model.eval()

    # ------------- Generate Finetuned Outputs ------------- #

    ft_pipe = pipeline(
        "image-text-to-text",
        model=model,  
        processor=processor,
        torch_dtype=torch.bfloat16,
    )

    # Optional inference tweaks
    ft_pipe.model.generation_config.do_sample = False
    ft_pipe.model.generation_config.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.padding_side = "left"

    ft_outputs = ft_pipe(
        text=test_data["messages"],
        images=test_data["image"],
        max_new_tokens=20,
        batch_size=4,
        return_full_text=False,
    )

    # The finetuned model's predictions
    ft_predictions = [postprocess(out, do_full_match=True) for out in ft_outputs]

    # -------- Evaluation -------- #

     # Define evaluation metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(predictions: list[int]) -> dict[str, float]:
        metrics = {}
        metrics.update(accuracy_metric.compute(
            predictions=predictions,
            references=ground_truth_labels,
        ))
        metrics.update(f1_metric.compute(
            predictions=predictions,
            references=ground_truth_labels,
            average="weighted",
        ))
        return metrics

    # employ evaluation metrics
    ft_metrics = compute_metrics(ft_predictions)
    print(f"Fine-tuned metrics: {ft_metrics}")

if __name__ == "__main__":
    main()

