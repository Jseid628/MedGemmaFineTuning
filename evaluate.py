# Will be evaluating the finetuned model here

from utils import format_data
from utils import load_model_and_processor
from datasets import load_dataset

# model, processor = load_model_and_processor()
# model.eval()

data = load_dataset("./patchcamelyon_subset")
print(data)