
from pathlib import Path
import shutil
import kagglehub

# Download the dataset using KaggleHub
dataset_path = Path(
    kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
)

print(f"Downloaded dataset to: {dataset_path}")

# Show the downloaded directory structure
for item in list(dataset_path.iterdir())[:20]:
    print(item.name, "DIR" if item.is_dir() else "FILE")

# Do not copy anything automatically until we verify
# the exact folder layout of your downloaded dataset.
