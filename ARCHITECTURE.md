# Image Detection Architecture

The project classifies still images as real or deepfake.

```text
Image file
    |
    v
Resize to 224x224 and ImageNet normalization
    |
    v
ImageNet-pretrained ConvNeXt-Tiny
    |
    v
Two-class classifier
    |
    v
REAL or FAKE prediction with confidence
```

## Training

`convnext_trainer.py` loads images from the paths in `config.yaml`, fine-tunes
ConvNeXt-Tiny, and saves the best Lightning checkpoint in `models/`.

## Inference

- `classify.py` classifies one image from the command line.
- `web-app.py` provides an image upload interface.
- `realeval.py` evaluates a folder of image samples.
- `inference/export_onnx.py` exports the image model to ONNX.
