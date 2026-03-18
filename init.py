import os, subprocess, zipfile
from pathlib import Path

ds_path = Path("./dataset/FGNET")

if not ds_path.exists():
    zip_path = Path("./fgnet-dataset.zip")
    subprocess.run([
        "curl", "-L", "-o", str(zip_path),
        "https://www.kaggle.com/api/v1/datasets/download/aiolapo/fgnet-dataset"
    ], check=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("./dataset")
    zip_path.unlink()

files = os.listdir(ds_path)
for file in files:
    sub_file = os.path.join(ds_path, str(file))
    for item in os.listdir(sub_file):
        if item != item.lower():
            old_name = os.path.join(sub_file, item)
            new_name = os.path.join(sub_file, item.lower())
            print(f"Renaming: {item} -> {item.lower()}")
            os.rename(old_name, new_name)

print("[init.py] - Done.")
