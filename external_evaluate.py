import argparse
import io
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from convnext_model import load_convnext_checkpoint
from image_preprocessing import crop_largest_face


class ExternalDeepfakeDataset(Dataset):
    def __init__(self, parquet_path, transform):
        self.data = pd.read_parquet(parquet_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_bytes = row["image"]["bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = crop_largest_face(image)
        return self.transform(image), int(row["label"])


def evaluate(parquet_path, checkpoint_path, batch_size, max_images):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = ExternalDeepfakeDataset(parquet_path, transform)
    if max_images:
        dataset.data = dataset.data.iloc[:max_images]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = load_convnext_checkpoint(checkpoint_path)
    model.eval()

    targets = []
    predictions = []
    probabilities = []

    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            # Dataset labels are converted to the project's convention:
            # 0 = Real, 1 = Fake.
            predictions.extend(preds.cpu().numpy().tolist())
            probabilities.extend(probs[:, 1].cpu().numpy().tolist())
            targets.extend((1 - labels.numpy()).tolist())

    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])

    print(f"Evaluated samples: {len(targets)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Real - precision: {precision[0]:.4f}, recall: {recall[0]:.4f}, f1: {f1[0]:.4f}")
    print(f"Fake - precision: {precision[1]:.4f}, recall: {recall[1]:.4f}, f1: {f1[1]:.4f}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print("           Real  Fake")
    print(f"Real       {matrix[0, 0]:5d} {matrix[0, 1]:5d}")
    print(f"Fake       {matrix[1, 0]:5d} {matrix[1, 1]:5d}")
    print("\nClassification report:")
    print(classification_report(
        targets,
        predictions,
        target_names=["Real", "Fake"],
        zero_division=0,
    ))

    results = pd.DataFrame({
        "true_label": ["Real" if value == 0 else "Fake" for value in targets],
        "predicted_label": ["Real" if value == 0 else "Fake" for value in predictions],
        "dataset_fake_probability": probabilities,
    })
    output_path = Path("external_evaluation_predictions.csv")
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        default="data/external_tests/deepfake-real-images-2/data/test-00000-of-00001.parquet",
    )
    parser.add_argument("--checkpoint", default="models/convnext_tiny_best.ckpt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()
    evaluate(args.parquet, args.checkpoint, args.batch_size, args.max_images)
