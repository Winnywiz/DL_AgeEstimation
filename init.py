import os
import shutil
import zipfile
import subprocess
from pathlib import Path
import kagglehub

ds_path = Path("./dataset/FGNET")

if not ds_path.exists():
    zip_path = Path("./fgnet-dataset.zip")
    print("[init] Downloading FGNET dataset...")
    subprocess.run([
        "curl", "-L", "-o", str(zip_path),
        "https://www.kaggle.com/api/v1/datasets/download/aiolapo/fgnet-dataset"
    ], check=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("./dataset")
    zip_path.unlink()
    print("[init] FGNET downloaded.")
else:
    print("[init] FGNET already exists, skipping download.")

for subject_dir in os.listdir(ds_path):
    subject_path = os.path.join(ds_path, str(subject_dir))
    if not os.path.isdir(subject_path):
        continue
    for item in os.listdir(subject_path):
        if item != item.lower():
            os.rename(
                os.path.join(subject_path, item),
                os.path.join(subject_path, item.lower())
            )
            print(f"  Renamed: {item} → {item.lower()}")

print("[init] FGNET done.\n")

output_base = Path("./UTKFace_organized")

age_bins = {
    "18-24"   : (18, 24),
    "25-39"   : (25, 39),
    "40-59"   : (40, 59),
    "60-plus" : (60, 116),
}

if not output_base.exists():
    print("[init] Downloading UTKFace dataset...")
    path = kagglehub.dataset_download("jangedoo/utkface-new")
    print(f"[init] Downloaded to: {path}")

    raw_data_path = os.path.join(path, "utkface_aligned_cropped", "UTKFace")

    for bin_name in age_bins:
        os.makedirs(output_base / bin_name, exist_ok=True)

    if not os.path.exists(raw_data_path):
        print(f"[init] Error: path does not exist: {raw_data_path}")
    else:
        all_files = os.listdir(raw_data_path)
        print(f"[init] Total files found: {len(all_files)}")
        count = 0
        for filename in all_files:
            if filename.lower().endswith(".jpg"):
                try:
                    age = int(filename.split("_")[0])
                    for bin_name, (low, high) in age_bins.items():
                        if low <= age <= high:
                            shutil.copy(
                                os.path.join(raw_data_path, filename),
                                output_base / bin_name / filename
                            )
                            count += 1
                            break
                except Exception:
                    continue
        print(f"[init] Organized {count} images into {output_base}")
else:
    print("[init] UTKFace already organized, skipping.")

print("\n" + "-" * 35)
print("UTKFace AGE RANGE DISTRIBUTION")
print("-" * 35)
distribution = {}
for folder in sorted(os.listdir(output_base)):
    folder_path = output_base / folder
    if folder_path.is_dir():
        count = len([f for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ".png"))])
        distribution[folder] = count
        print(f"  {folder:10} : {count:,} images")
print("-" * 35)
print("[init] Done.")
