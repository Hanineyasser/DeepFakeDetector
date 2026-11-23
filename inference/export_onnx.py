import torch
from convnext_model import load_convnext_checkpoint

model = load_convnext_checkpoint("models/convnext_tiny_best.ckpt")

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "deepfake_model.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size"}})

print("✅ Exported to deepfake_model.onnx")
