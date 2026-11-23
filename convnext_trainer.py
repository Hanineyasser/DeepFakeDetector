import torch
import yaml
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from datasets.hybrid_loader import HybridDeepfakeDataset
from lightning_modules.detector import DeepfakeDetector
from convnext_model import build_convnext_tiny
from image_preprocessing import crop_largest_face


def main():
    with open("config.yaml") as config_file:
        cfg = yaml.safe_load(config_file)

    train_transform = transforms.Compose([
        transforms.Lambda(crop_largest_face),
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.25),
        transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=0.5)], p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    val_transform = transforms.Compose([
        transforms.Lambda(crop_largest_face),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_sources = [(path, None) for path in cfg["train_paths"]]
    val_sources = [(path, None) for path in cfg["val_paths"]]
    train_dataset = HybridDeepfakeDataset(train_sources, transform=train_transform)
    val_dataset = HybridDeepfakeDataset(val_sources, transform=val_transform)

    if not train_dataset or not val_dataset:
        raise ValueError(
            "Training and validation datasets must each contain at least one image."
        )

    real_count = sum(label == 0 for label in train_dataset.labels)
    fake_count = sum(label == 1 for label in train_dataset.labels)
    class_weights = [
        len(train_dataset) / (2.0 * real_count),
        len(train_dataset) / (2.0 * fake_count),
    ]
    sample_weights = [
        class_weights[label] for label in train_dataset.labels
    ]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        sampler=sampler,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    backbone = build_convnext_tiny(pretrained=True)
    model = DeepfakeDetector(
        backbone,
        lr=cfg["lr"],
        class_weights=torch.tensor(class_weights, dtype=torch.float),
    )

    checkpoint = ModelCheckpoint(
        monitor=cfg.get("monitor_metric", "val_loss"),
        dirpath="models",
        filename="convnext_tiny_best",
        save_top_k=cfg.get("save_top_k", 1),
        save_last=True,
        every_n_train_steps=cfg.get("checkpoint_every_n_steps", 500),
        mode="max" if cfg.get("monitor_metric") == "val_balanced_acc" else "min",
    )
    early_stop = EarlyStopping(
        monitor=cfg.get("monitor_metric", "val_loss"),
        patience=cfg.get("early_stopping_patience", 3),
        mode="max" if cfg.get("monitor_metric") == "val_balanced_acc" else "min",
    )

    loggers = []
    version = None
    try:
        tb_logger = TensorBoardLogger(save_dir=".", name="lightning_logs")
        version = tb_logger.version
        loggers.append(tb_logger)
    except ModuleNotFoundError:
        pass
    loggers.append(CSVLogger(save_dir=".", name="lightning_logs", version=version))

    trainer = pl.Trainer(
        max_epochs=cfg["num_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=loggers,
        callbacks=[checkpoint, early_stop],
        enable_progress_bar=True,
        log_every_n_steps=cfg.get("log_every_n_steps", 1),
    )

    resume_path = cfg.get("resume_from_checkpoint")
    if resume_path:
        resume_path = str(Path(resume_path).expanduser())
        if not Path(resume_path).is_file():
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {resume_path}"
            )
        print(f"Resuming training from {resume_path}")

    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=resume_path,
    )


if __name__ == "__main__":
    main()
