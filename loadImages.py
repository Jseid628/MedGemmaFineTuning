from datasets import load_dataset

dataset = load_dataset("1aurent/PatchCamelyon")

# Optional: subset to first 10,000 samples
dataset = dataset["train"].shuffle(seed=42).select(range(10000))

# Inspect
example = dataset[0]
image = example['image']

image.save("sample_image.png")
print("Image saved to sample_image.png")

print(dataset)
print(dataset.features)