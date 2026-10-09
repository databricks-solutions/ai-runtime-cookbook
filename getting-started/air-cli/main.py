import torch

print("Hello from Python on AI Runtime")
print(f"GPU: {torch.cuda.get_device_name(0)}")
