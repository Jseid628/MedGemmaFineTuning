from huggingface_hub import HfApi

api = HfApi()

# Replace with the full model ID if different
model_id="google/medgemma-4b-it"

# This will raise an error if you don't have access
model_info = api.model_info(model_id)

print(f"✅ You have access to: {model_info.modelId}")
print(f"📦 Model size: {model_info.safetensors.get('total') if model_info.safetensors else 'unknown'} bytes")
