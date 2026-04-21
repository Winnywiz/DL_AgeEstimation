import os
import torch
import torch.nn as nn
from datetime import datetime
from torchvision import datasets
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter

from model import ResNet50_base
from utils import train_one_epoch, evaluate, plot_predictions

DATA_DIR      = "./UTKFace_organized"
CKPT_PATH     = "./checkpoints/resnet50_base.pth"
LOG_DIR       = "./runs/resnet50_baseline"
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
EPOCHS        = 10
NUM_CLASSES   = 4
SEED          = 42

if __name__ == '__main__':
    os.makedirs("./checkpoints", exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    train_trans = T.Compose([
        T.Resize((224, 224)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_trans = T.Compose([
        T.Resize((224, 224)),
        T.CenterCrop(224),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    base_train_ds = datasets.ImageFolder(root=DATA_DIR, transform=train_trans)
    base_test_ds  = datasets.ImageFolder(root=DATA_DIR, transform=test_trans)

    print(f"Classes : {base_train_ds.classes}")
    print(f"Total   : {len(base_train_ds):,} images")

    total_size = len(base_train_ds)
    train_size = int(0.8 * total_size)
    val_size   = int(0.1 * total_size)
    test_size  = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(SEED)
    train_idx, val_idx, test_idx = random_split(
        range(total_size), [train_size, val_size, test_size], generator=generator
    )

    train_ds = Subset(base_train_ds, train_idx)
    val_ds   = Subset(base_test_ds,  val_idx)
    test_ds  = Subset(base_test_ds,  test_idx)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    model     = ResNet50_base(num_classes=NUM_CLASSES).to(device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total_p:,} params")

    batch_x, batch_y = next(iter(train_dl))
    with torch.no_grad():
        y_hat  = model(batch_x.to(device))
        y_pred = y_hat.argmax(dim=1).cpu()

    plot_predictions(
        batch_x, batch_y, y_pred,
        class_names=base_train_ds.classes,
        save_path="./resnet_baseline_sanity.jpg"
    )

    writer     = SummaryWriter(f"{LOG_DIR}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    best_vloss = float("inf")

    print(f"\nStarting baseline training — {EPOCHS} epochs\n")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1} / {EPOCHS}")
        train_one_epoch(train_dl, model, loss_fn, optimizer, epoch, device, writer)

        train_loss, train_acc, train_f1 = evaluate(train_dl, model, loss_fn, device)
        val_loss,   val_acc,   val_f1   = evaluate(val_dl,   model, loss_fn, device)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("Acc/train",  train_acc,  epoch)
        writer.add_scalar("Acc/val",    val_acc,    epoch)
        writer.add_scalar("F1/train",   train_f1,   epoch)
        writer.add_scalar("F1/val",     val_f1,     epoch)

        print(f"  Train → loss={train_loss:.4f}  acc={100*train_acc:.1f}%  f1={100*train_f1:.1f}%")
        print(f"  Val   → loss={val_loss:.4f}  acc={100*val_acc:.1f}%  f1={100*val_f1:.1f}%")

        if val_loss < best_vloss:
            best_vloss = val_loss
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"  ✓ Saved best model → {CKPT_PATH}")

    writer.close()
    print("\nTraining Done!")

    print("\nLoading best checkpoint for final evaluation...")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))

    train_loss, train_acc, train_f1 = evaluate(train_dl, model, loss_fn, device)
    val_loss,   val_acc,   val_f1   = evaluate(val_dl,   model, loss_fn, device)
    test_loss,  test_acc,  test_f1  = evaluate(test_dl,  model, loss_fn, device)

    print("=" * 50)
    print("Final Evaluation — ResNet-50 Baseline")
    print("=" * 50)
    print(f"  Train → loss={train_loss:.4f}  acc={100*train_acc:.1f}%  f1={100*train_f1:.1f}%")
    print(f"  Val   → loss={val_loss:.4f}  acc={100*val_acc:.1f}%  f1={100*val_f1:.1f}%")
    print(f"  Test  → loss={test_loss:.4f}  acc={100*test_acc:.1f}%  f1={100*test_f1:.1f}%")
    print("=" * 50)
