import torch
from Dataset import Dataset
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader, random_split

MAX_AGE = 69
learning_rate = 1e-3
batch_size = 64
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

ds_path     = './dataset/FGNET'
ds          = Dataset(path=ds_path)

train_transform = transforms_v2.Compose([
    transforms_v2.Resize((128, 128)),
    transforms_v2.RandomHorizontalFlip(),
    transforms_v2.RandomRotation(degrees=10),
    transforms_v2.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms_v2.RandomGrayscale(p=0.1),
    transforms_v2.ToImage(),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms_v2.Compose([
    transforms_v2.Resize((128, 128)),
    transforms_v2.ToImage(),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

valid_samples   = int(0.1 * len(ds))
test_samples    = int(0.2 * len(ds))
train_samples   = int(len(ds) - test_samples - valid_samples)
train_ds, test_ds, valid_ds = random_split(ds, [train_samples, test_samples, valid_samples])

train_ds.dataset.transform = train_transform
valid_ds.dataset.transform = test_transform
test_ds.dataset.transform = test_transform

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

print(device)
print(train_samples, test_samples, valid_samples, (valid_samples + test_samples + train_samples) == len(ds))
print(len(train_ds))