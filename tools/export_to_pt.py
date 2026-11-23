import torch
from convnext_model import load_convnext_checkpoint

ckpt_path = "models/convnext_tiny_best.ckpt"
pt_output = "models/convnext_tiny_best.pt"

model = load_convnext_checkpoint(ckpt_path)

torch.save(model.state_dict(), pt_output)

print(f"✅ Exported PyTorch model to {pt_output}")
