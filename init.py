import os, subprocess

ds_path = "./dataset/FGNET"

if not os.path.exists(ds_path):
    subprocess.run(["bash", "./scripts/dataset.sh"])

files = os.listdir(ds_path)
for file in files:
    sub_file = os.path.join(ds_path, file)
    for item in os.listdir(sub_file):
        old_name = os.path.join(sub_file, item)
        new_name = os.path.join(sub_file, item.lower())
        if old_name != new_name:
            print(f"Renaming: {item} -> {item.lower()}")
            os.rename(old_name, new_name)
print("[init.py] - Done.")
