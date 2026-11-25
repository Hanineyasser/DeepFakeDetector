import os
from torch.utils.data import Dataset
from PIL import Image

class HybridDeepfakeDataset(Dataset):
    def __init__(self, sources, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        class_map = {'real': 0, 'fake': 1}
        valid_exts = ('.jpg', '.jpeg', '.png', '.webp')

        for path, override_label in sources:
            if not os.path.isdir(path):
                continue
            if override_label is None:
                for entry in os.scandir(path):
                    if not entry.is_dir():
                        continue
                    mapped = class_map.get(entry.name.lower())
                    if mapped is None:
                        continue
                    for root, _, files in os.walk(entry.path):
                        for fname in files:
                            if fname.lower().endswith(valid_exts):
                                self.image_paths.append(os.path.join(root, fname))
                                self.labels.append(mapped)
            else:
                for root, _, files in os.walk(path):
                    for fname in files:
                        if fname.lower().endswith(valid_exts):
                            self.image_paths.append(os.path.join(root, fname))
                            self.labels.append(override_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
