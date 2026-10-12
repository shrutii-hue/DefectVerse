"""Training script to fit PatchCore on normal industrial inspection samples."""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from src.models.patchcore import PatchCore


class MVTecTrainDataset(Dataset):
    """Dataset for defect-free normal training images."""

    def __init__(self, image_paths, transform=None):
        self.image_paths = sorted(image_paths)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = str(self.image_paths[idx])
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, path



def train_patchcore(
    data_dir: str = "data/bottle",
    output_path: str = "outputs/models/patchcore_bottle.pt",
    backbone: str = "resnet18",
    coreset_ratio: float = 0.1,
    batch_size: int = 8,
    device: str = None,
) -> str:
    """Train and persist PatchCore model."""
    start_time = time.time()
    data_path = Path(data_dir).resolve()
    train_dir = data_path / "train" / "good"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found at: {train_dir}\n"
            f"Please run 'python src/data/download_dataset.py' or 'python src/data/download_dataset.py --mock' first."
        )

    # Collect training images
    image_paths = list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.jpeg"))
    if not image_paths:
        raise ValueError(f"No image files found in {train_dir}")

    print("=" * 65)
    print(f"       DEFECTVERSE - PATCHCORE MODEL TRAINING")
    print("=" * 65)
    print(f"Dataset category path : {train_dir}")
    print(f"Normal training images: {len(image_paths)}")
    print(f"Feature Backbone      : {backbone}")
    print(f"Coreset ratio         : {coreset_ratio * 100:.1f}%")
    print(f"Batch size            : {batch_size}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Compute device        : {device.upper()}")
    print("-" * 65)

    # Initialize PatchCore model
    model = PatchCore(
        backbone_name=backbone,
        layers=("layer2", "layer3"),
        device=device,
        coreset_sampling_ratio=coreset_ratio,
    )

    # Create dataset and dataloader
    dataset = MVTecTrainDataset(image_paths, transform=model.transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Cross-platform Windows safe default
        pin_memory=(device == "cuda"),
    )

    # Fit memory bank
    print("\n[Step 1/2] Extracting patch representations and building memory bank...")
    model.fit(dataloader)

    # Save artifact
    print(f"\n[Step 2/2] Persisting model artifact...")
    dest_path = Path(output_path)
    saved_file = model.save(dest_path)
    file_size_mb = os.path.getsize(saved_file) / (1024 * 1024)

    elapsed = time.time() - start_time
    print("=" * 65)
    print(f"TRAINING COMPLETED SUCCESSFULLY in {elapsed:.2f}s")
    print(f"Saved artifact : {saved_file} ({file_size_mb:.2f} MB)")
    print(f"Memory bank    : {model.memory_bank.shape[0]} patches x {model.memory_bank.shape[1]} dims")
    print(f"Calibrated Thresh: {model.threshold:.4f}")
    print("=" * 65)
    return saved_file


def main():
    parser = argparse.ArgumentParser(
        description="Fit PatchCore memory bank on normal industrial images."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/bottle",
        help="Path to dataset category directory (default: data/bottle)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/models/patchcore_bottle.pt",
        help="Output checkpoint file path (default: outputs/models/patchcore_bottle.pt)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "wide_resnet50_2"],
        help="Pretrained backbone architecture (default: resnet18)",
    )
    parser.add_argument(
        "--coreset_ratio",
        type=float,
        default=0.1,
        help="Coreset subsampling fraction (default: 0.1 / 10%%)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for feature extraction (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda' or 'cpu'). Default: auto-detect",
    )
    args = parser.parse_args()

    train_patchcore(
        data_dir=args.data_dir,
        output_path=args.output_path,
        backbone=args.backbone,
        coreset_ratio=args.coreset_ratio,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
