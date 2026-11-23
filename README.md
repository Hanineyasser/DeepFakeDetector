# ðŸ§  Deepfake Detection System

A state-of-the-art deepfake detection system built with PyTorch and ConvNeXt-Tiny, featuring a user-friendly web interface for real-time image analysis.

## âš™ï¸ Created By

- ðŸ‘¨â€ðŸ’» [T RAHUL SINGH](https://github.com/TRahulsingh)
- ðŸ§‘â€ðŸ’» [Mallikarjun Macherla](https://github.com/Mallikarjun-Macherla)
- ðŸ§‘â€ðŸ’» [Sainath](https://github.com/sainathch45/)

---

## ðŸŒŸ Features

- **Deep Learning Model**: ConvNeXt-Tiny architecture fine-tuned for deepfake detection
- **Image Support**: Analyze images (.jpg, .jpeg, .png)
- **Web Interface**: Interactive Gradio-based web application for easy testing
- **Real-time Analysis**: Classify uploaded images with confidence scores
- **Training Pipeline**: Complete PyTorch Lightning training infrastructure
- **Model Export**: Support for PyTorch (.pt) and ONNX format exports

## ðŸ“ System Architecture

For detailed system architecture diagrams, data flow, and component interactions, see [ARCHITECTURE.md](ARCHITECTURE.md).

## ðŸš€ Quick Start

### ðŸ–¼ï¸ Image-based Datasets

#### **140k Real and Fake Faces**
- **Description**: Large collection of real and AI-generated face images
- **Size**: ~140,000 images
- **Source**: StyleGAN-generated faces vs real faces
- **Download**: [Kaggle Dataset](https://www.kaggle.com/xhlulu/140k-real-and-fake-faces)
- **Usage**: Perfect for image-based deepfake detection training

#### **CelebA-HQ**
- **Description**: High-quality celebrity face dataset
- **Size**: 30,000 high-resolution images
- **Quality**: 1024Ã—1024 resolution
- **Download**: [GitHub Repository](https://github.com/tkarras/progressive_growing_of_gans)
- **Usage**: Real face examples for training

### ðŸ”§ Dataset Preparation

#### Option 1: Download Pre-processed Datasets
1. Download your chosen dataset from the links above
2. Extract to the `data/` folder
3. Organize as shown in the training section below

#### Option 2: Use Dataset Preparation Tools
Use our built-in tools to prepare datasets. Edit the source/destination paths inside each script before running:

```bash
# Edit source & dest paths in the script, then run:

# Split an existing image dataset into 80/20 train/validation
# Edit source_dataset & destination paths in the script, then run:
python tools/split_train_val.py

```

### ðŸ“‹ Dataset Recommendations

- **For Beginners**: Start with **140k Real and Fake Faces** (image-based, easy to work with)
- **For Research**: Use **FaceForensics++** (comprehensive, multiple manipulation types)
- **For Production**: Combine **DFDC** + **Celeb-DF** (large scale, diverse)
- **For High-Quality Testing**: Use **Celeb-DF v2** (challenging, high-quality deepfakes)

### âš ï¸ Dataset Usage Notes

- **Ethical Use**: These datasets are for research purposes only
- **Legal Compliance**: Ensure compliance with dataset licenses and terms of use
- **Privacy**: Respect privacy rights of individuals in the datasets
- **Citation**: Properly cite the original dataset papers when publishing research

## ðŸ‹ï¸ Training

### Dataset Structure

Organize your training data in the `data` folder as follows:
```
data/
â”œâ”€â”€ train/
â”‚   â”œâ”€â”€ real/
â”‚   â”‚   â”œâ”€â”€ image1.jpg
â”‚   â”‚   â””â”€â”€ image2.jpg
â”‚   â””â”€â”€ fake/
â”‚       â”œâ”€â”€ fake1.jpg
â”‚       â””â”€â”€ fake2.jpg
â””â”€â”€ validation/
    â”œâ”€â”€ real/
    â””â”€â”€ fake/
```

### Configuration

Update `config.yaml` with your dataset paths:

```yaml
train_paths:
  - data/train

val_paths:
  - data/validation

lr: 0.0001
batch_size: 4
num_epochs: 10
```

### Start Training

```bash
python main_trainer.py
```

The training will:
- Use PyTorch Lightning for efficient training
- Save best model based on validation loss
- Save a recovery checkpoint every 500 training batches
- Log metrics to TensorBoard
- Apply early stopping to prevent overfitting

### Resume Training

The recovery checkpoint is saved to `models/last.ckpt`. If training is interrupted,
set `resume_from_checkpoint` in `config.yaml` and start training again:

```yaml
resume_from_checkpoint: "models/last.ckpt"
```

```bash
python main_trainer.py
```

Set `resume_from_checkpoint: null` to start from epoch 0. The save frequency can
be adjusted with `checkpoint_every_n_steps`.

### Monitor Training

View training progress with TensorBoard:

```bash
tensorboard --logdir lightning_logs
```

## ðŸ“ Project Structure

```
â”œâ”€â”€ web-app.py                    # Main web application
â”œâ”€â”€ main_trainer.py               # Primary training script
â”œâ”€â”€ classify.py                   # Image classification utility
â”œâ”€â”€ realeval.py                   # Real-world evaluation script
â”œâ”€â”€ config.yaml                   # Training configuration
â”œâ”€â”€ requirements.txt              # Python dependencies
â”œâ”€â”€ README.md                     # Project documentation
â”œâ”€â”€ ARCHITECTURE.md               # System architecture & design
â”œâ”€â”€ LICENSE                       # MIT License
â”œâ”€â”€ .gitignore                    # Git ignore rules
â”œâ”€â”€ data/                         # Dataset storage (not tracked by git)
â”‚   â”œâ”€â”€ train/                    # Training data
â”‚   â””â”€â”€ validation/               # Validation data
â”œâ”€â”€ datasets/
â”‚   â””â”€â”€ hybrid_loader.py          # Custom dataset loader
â”œâ”€â”€ lightning_modules/
â”‚   â””â”€â”€ detector.py               # PyTorch Lightning module
â”œâ”€â”€ models/
â”‚   â””â”€â”€ convnext_tiny_best.ckpt          # Trained model weights
â”œâ”€â”€ tools/                        # Dataset preparation utilities
â”‚   â”œâ”€â”€ export_to_pt.py           # .ckpt â†’ .pt model converter
â”‚   â”œâ”€â”€ split_train_val.py        # 80/20 train/val splitter
â””â”€â”€ inference/
    â”œâ”€â”€ export_onnx.py            # ONNX export
```

## ðŸ› ï¸ Model Architecture

- **Backbone**: ConvNeXt-Tiny (pre-trained on ImageNet)
- **Classifier**: Custom 2-class classifier with dropout (0.4)
- **Input Size**: 224x224 RGB images
- **Output**: Binary classification (Real/Fake) with confidence scores

## ðŸ“Š Performance

- **Inference Speed**: Real-time on GPU, ~200ms per image on CPU
- **Input Support**: Images (.jpg, .jpeg, .png)
- **Robustness**: Tested with Gaussian blur and JPEG compression noise simulation (`realeval.py`)

> **Note**: Accuracy metrics depend on your training dataset. Monitor `val_loss` and `val_acc` via TensorBoard during training.

## ðŸ”§ Advanced Usage

### Export to ONNX

Convert PyTorch model to ONNX format:

```bash
python inference/export_onnx.py
```

### Batch Evaluation

Evaluate a folder of real-world samples with optional noise simulation:

```bash
# Place test images in realworld_samples/ folder, then run:
python realeval.py
```

### Export Checkpoint to PyTorch

Convert a Lightning `.ckpt` to a standalone `.pt` file:

```bash
# Edit ckpt_path and pt_output in the script, then run:
python tools/export_to_pt.py
```

## ðŸ¤ Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ðŸ™ Acknowledgments

- ConvNeXt architecture by Meta AI
- PyTorch Lightning for training infrastructure
- Gradio for web interface framework
- The research community for deepfake detection advances

---

## ðŸ“„ License

This project is licensed under the **MIT License**.

---

â­ **Star this repository if you found it helpful!**


