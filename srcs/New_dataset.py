import os
import shutil

import kagglehub

path = kagglehub.dataset_download("jangedoo/utkface-new")

print("Path to dataset files:", path)
# Path should be --> r"C:\Users\User\.cache\kagglehub\datasets\jangedoo\utkface-new\versions\1"

# Organise the UTKFace dataset into age bins

raw_data_path = os.path.join(path, "utkface_aligned_cropped", "UTKFace") 
output_base = r"../UTKFace_organized"

age_bins = {
    "18-24": (18, 24),
    "25-39": (25, 39),
    "40-59": (40, 59),
    "60-plus": (60, 116)
}

for bin_name in age_bins:
    os.makedirs(os.path.join(output_base, bin_name), exist_ok=True)

print("Starting organization...")

if not os.path.exists(raw_data_path):
    print(f"Error: Path does not exist: {raw_data_path}")
else:
    all_files = os.listdir(raw_data_path)
    print(f"Total files found in raw folder: {len(all_files)}")
    
    count = 0
    for filename in all_files:
        if filename.lower().endswith(".jpg"):
            try:
                age = int(filename.split('_')[0])
                
                for bin_name, (low, high) in age_bins.items():
                    if low <= age <= high:
                        shutil.copy(
                            os.path.join(raw_data_path, filename),
                            os.path.join(output_base, bin_name, filename)
                        )
                        count += 1
                        break
            except Exception as e:
                continue

    print(f"Successfully organized {count} images into {output_base}")