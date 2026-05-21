"""Download the Rock-Paper-Scissors dataset (Laurence Moroney / TF)."""

import subprocess
import zipfile
from pathlib import Path

TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/rps.zip"
TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/rps-test-set.zip"

DATA_DIR = Path("data")


def download(url: str, dest: Path) -> None:
    zip_path = DATA_DIR / (dest.name + ".zip")
    if not zip_path.exists():
        print(f"Downloading {dest.name} ...")
        subprocess.run(["curl", "-L", "-o", str(zip_path), "--progress-bar", url], check=True)
    else:
        print(f"Already downloaded: {zip_path}")

    if not dest.exists():
        print(f"Extracting to {dest} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
        print("Done.")
    else:
        print(f"Already extracted: {dest}")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    download(TRAIN_URL, DATA_DIR / "rps")
    download(TEST_URL, DATA_DIR / "rps-test-set")

    print("\nDataset structure:")
    for split in ("rps", "rps-test-set"):
        split_dir = DATA_DIR / split
        if split_dir.exists():
            for cls_dir in sorted(split_dir.iterdir()):
                if cls_dir.is_dir():
                    count = sum(1 for _ in cls_dir.glob("*.png"))
                    print(f"  {split}/{cls_dir.name}: {count} images")
