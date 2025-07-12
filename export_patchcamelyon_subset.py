from datasets import load_dataset
from pathlib import Path
from PIL import Image

# Step 1: Load the dataset
dataset = load_dataset("1aurent/PatchCamelyon")
subset = dataset["train"].shuffle(seed=42).select(range(10000))

# Step 2: Create output folders
output_root = Path("patchcamelyon_subset")
(output_root / "tumor").mkdir(parents=True, exist_ok=True)
(output_root / "normal").mkdir(parents=True, exist_ok=True)

# Step 3: Loop through the subset and export images
for idx, example in enumerate(subset):
    image: Image.Image = example["image"]
    label = example["label"]
    
    label_str = "tumor" if label == 1 else "normal"
    filename = f"img_{idx:05d}.png"
    filepath = output_root / label_str / filename

    image.save(filepath)

    if idx % 1000 == 0:
        print(f"Saved {idx} images...")

print("Done exporting images.")

