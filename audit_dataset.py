import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root):
    root = Path(root)
    hashes = defaultdict(list)
    formats = defaultdict(int)
    sizes = defaultdict(int)
    counts = defaultdict(int)
    corrupt = []

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative = path.relative_to(root)
        label = relative.parts[0].lower() if relative.parts else "unknown"
        counts[label] += 1

        try:
            with Image.open(path) as image:
                image.load()
                formats[image.format] += 1
                sizes[image.size] += 1
        except Exception as error:
            corrupt.append((str(relative), str(error)))
            continue

        hashes[sha256(path)].append(str(relative))

    duplicates = {
        image_hash: paths
        for image_hash, paths in hashes.items()
        if len(paths) > 1
    }

    print(f"Root: {root}")
    print(f"Images: {sum(counts.values())}")
    print("Class counts:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    print(f"Corrupt or unreadable: {len(corrupt)}")
    print(f"Duplicate files: {sum(len(paths) - 1 for paths in duplicates.values())}")
    print(f"Unique duplicate groups: {len(duplicates)}")
    print("Top image sizes:")
    for (width, height), count in sorted(sizes.items(), key=lambda item: item[1], reverse=True)[:10]:
        print(f"  {width}x{height}: {count}")
    print("Image formats:")
    for image_format, count in sorted(formats.items(), key=lambda item: item[1], reverse=True):
        print(f"  {image_format}: {count}")

    for label, count in counts.items():
        print(f"{label}_count={count}")

    if corrupt:
        print("Corrupt examples:")
        for path, error in corrupt[:20]:
            print(f"  {path}: {error}")


def find_duplicate_hashes(root):
    root = Path(root)
    hashes = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            hashes[sha256(path)].append(str(path))
    return {
        image_hash: paths
        for image_hash, paths in hashes.items()
        if len(paths) > 1
    }


def audit_leakage(train_root, val_root, test_root=None):
    roots = [
        ("train", Path(train_root)),
        ("validation", Path(val_root)),
    ]
    if test_root:
        roots.append(("test", Path(test_root)))

    hashes_by_source = {}
    for name, root in roots:
        hashes = defaultdict(list)
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                hashes[sha256(path)].append(str(path))
        hashes_by_source[name] = hashes

    leaks = []
    for left_index, (left_name, left_hashes) in enumerate(hashes_by_source.items()):
        for right_name, right_hashes in list(hashes_by_source.items())[left_index + 1:]:
            duplicate_hashes = set(left_hashes) & set(right_hashes)
            for image_hash in duplicate_hashes:
                leaks.append((left_name, left_hashes[image_hash], right_name, right_hashes[image_hash]))

    print("Cross-split duplicate audit:")
    if leaks:
        for left_name, left_paths, right_name, right_paths in leaks:
            print(f"{left_name} <-> {right_name}")
            for left_path in left_paths:
                print(f"  {left_path}")
            for right_path in right_paths:
                print(f"  {right_path}")
        print(f"Leakage groups found: {len(leaks)}")
        return False

    print("No duplicate image files found across the requested splits.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?")
    parser.add_argument("--validation")
    parser.add_argument("--test")
    args = parser.parse_args()
    if args.validation:
        if not args.root:
            parser.error("--validation requires a train root")
        audit_leakage(args.root, args.validation, args.test)
    elif args.root:
        audit(args.root)
