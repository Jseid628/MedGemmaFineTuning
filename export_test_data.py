# -------------- exporting test data -------------- #

from datasets import load_dataset
from pathlib import Path
from PIL import Image

# Load the dataset
dataset = load_dataset("1aurent/PatchCamelyon")
patchcamelyon_test = dataset["test"].shuffle(seed = 42).select(range(2000))

test_output_root = Path("patchcamelyon_test")
(test_output_root / "tumor").mkdir(parents=True, exist_ok=True)
(test_output_root / "normal").mkdir(parents=True, exist_ok=True)

# Test

# Loop through the test subset and export images
for idx, example in enumerate(patchcamelyon_test):
    image: Image.Image = example["image"]
    label = example["label"]
    
    label_str = "tumor" if label == 1 else "normal"
    filename = f"img_{idx:05d}.png"
    filepath = test_output_root / label_str / filename

    image.save(filepath)

    if idx % 1000 == 0:
        print(f"Saved {idx} images...")

print("Done exporting images.")