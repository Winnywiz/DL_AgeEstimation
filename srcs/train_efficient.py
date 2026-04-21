import os
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score

from model import EfficientNetB0
from utils import train_one_epoch, test, plot_predictions

DATA_DIR      = "./UTKFace_organized"
CKPT_P1       = "./checkpoints/efficientnet_b0_vloss.pth"
CKPT_P2       = "./checkpoints/efficientnet_b0_finetuned.pth"
LOG_DIR       = "./runs/efficientnet_b0_transfer"
BATCH_SIZE    = 64
LEARNING_RATE = 5e-4
EPOCHS_P1     = 15
EPOCHS_P2     = 30
PATIENCE      = 15
NUM_CLASSES   = 4
SEED          = 42


def evaluate_1_off_accuracy(preds, trues):
    """Accuracy allowing ±1 class error (adjacent age groups count as correct)."""
    if not isinstance(preds, torch.Tensor):
        preds = torch.tensor(preds)
    if not isinstance(trues, torch.Tensor):
        trues = torch.tensor(trues)
    diff = torch.abs(preds - trues)
    return torch.sum(diff <= 1).item() / len(trues)


if __name__ == "__main__":
    os.makedirs("./checkpoints", exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    train_trans = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=5),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_trans = T.Compose([
        T.Resize((224, 224)),
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

    model, _ = EfficientNetB0(num_classes=NUM_CLASSES, freeze_backbone=True)
    model     = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total_p:,} params")

    batch_x, batch_y = next(iter(train_dl))
    with torch.no_grad():
        y_hat  = model(batch_x.to(device))
        y_pred = y_hat.argmax(dim=1).cpu()
    plot_predictions(batch_x, batch_y, y_pred, base_train_ds.classes, save_path="./efficient_sanity_check.jpg")

    class_weights = torch.tensor([1.89, 1.0, 1.45, 1.88]).to(device)
    loss_fn       = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer     = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    writer = SummaryWriter(f"{LOG_DIR}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    # ─────────────────────────────────────────────────
    # Phase 1: Head-Only Training
    # ─────────────────────────────────────────────────
    best_vloss = float("inf")

    if not os.path.exists(CKPT_P1):
        print(f"\nPhase 1: Head-only training — {EPOCHS_P1} epochs\n")
        for epoch in range(EPOCHS_P1):
            print(f"Epoch {epoch+1} / {EPOCHS_P1}")
            train_one_epoch(train_dl, model, loss_fn, optimizer, epoch, device, writer)

            val_loss, val_preds, val_trues = test(val_dl, model, loss_fn, device)
            val_acc  = multiclass_accuracy(val_preds, val_trues).item()
            val_f1   = multiclass_f1_score(val_preds, val_trues).item()
            val_1off = evaluate_1_off_accuracy(val_preds, val_trues)

            writer.add_scalar("Loss/val",    val_loss, epoch)
            writer.add_scalar("Acc/val",     val_acc,  epoch)
            writer.add_scalar("1OffAcc/val", val_1off, epoch)
            writer.add_scalar("F1/val",      val_f1,   epoch)

            print(f"  Val → loss={val_loss:.4f}  acc={100*val_acc:.1f}%  f1={100*val_f1:.1f}%  1-off={100*val_1off:.1f}%")

            if val_loss < best_vloss:
                best_vloss = val_loss
                torch.save(model.state_dict(), CKPT_P1)
                print(f"  ✓ Saved → {CKPT_P1}")
        print("Phase 1 Done!")
    else:
        print(f"Phase 1 checkpoint found — skipping. Delete {CKPT_P1} to retrain.")

    # ─────────────────────────────────────────────────
    # Phase 1 Evaluation
    # ─────────────────────────────────────────────────
    model_best, _ = EfficientNetB0(num_classes=NUM_CLASSES, freeze_backbone=False)
    model_best     = model_best.to(device)
    model_best.load_state_dict(torch.load(CKPT_P1, map_location=device, weights_only=True))
    model_best.eval()

    def full_eval(dl, tag):
        loss, preds, trues = test(dl, model_best, loss_fn, device)
        acc  = multiclass_accuracy(preds, trues).item()
        f1   = multiclass_f1_score(preds, trues).item()
        off1 = evaluate_1_off_accuracy(preds, trues)
        print(f"  {tag} → loss={loss:.4f}  acc={100*acc:.1f}%  f1={100*f1:.1f}%  1-off={100*off1:.1f}%")

    print("\n" + "=" * 50)
    print("Phase 1 Evaluation")
    print("=" * 50)
    full_eval(train_dl, "Train")
    full_eval(val_dl,   "Val  ")
    full_eval(test_dl,  "Test ")
    print("=" * 50)

    # ─────────────────────────────────────────────────
    # Phase 2: Full Fine-Tuning (Differential LR)
    # ─────────────────────────────────────────────────
    print("\n--- Phase 2: Fine-Tuning ---\n")
    model_best.train()

    for param in model_best.parameters():
        param.requires_grad = True

    optimizer_ft = torch.optim.AdamW([
        {"params": model_best.features.parameters(),   "lr": 1e-6},
        {"params": model_best.classifier.parameters(), "lr": 1e-4},
    ], weight_decay=1e-2)

    scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=EPOCHS_P2)
    best_vloss_ft = float("inf")
    counter       = 0

    for epoch in range(EPOCHS_P2):
        lr_head = optimizer_ft.param_groups[-1]["lr"]
        print(f"Fine-Tune Epoch {epoch+1} / {EPOCHS_P2}  (lr_head={lr_head:.2e})")

        train_one_epoch(train_dl, model_best, loss_fn, optimizer_ft, epoch, device, writer)
        scheduler.step()

        val_loss, val_preds, val_trues = test(val_dl, model_best, loss_fn, device)
        val_acc  = multiclass_accuracy(val_preds, val_trues).item()
        val_f1   = multiclass_f1_score(val_preds, val_trues).item()
        val_1off = evaluate_1_off_accuracy(val_preds, val_trues)

        writer.add_scalar("Loss/val_ft",     val_loss, epoch)
        writer.add_scalar("Acc/val_ft",      val_acc,  epoch)
        writer.add_scalar("1OffAcc/val_ft",  val_1off, epoch)
        writer.add_scalar("F1/val_ft",       val_f1,   epoch)

        print(f"  Val → loss={val_loss:.4f}  acc={100*val_acc:.1f}%  f1={100*val_f1:.1f}%  1-off={100*val_1off:.1f}%")

        if val_loss < best_vloss_ft:
            best_vloss_ft = val_loss
            torch.save({
                "epoch"              : epoch + 1,
                "model_state_dict"   : model_best.state_dict(),
                "optimizer_state_dict": optimizer_ft.state_dict(),
                "loss"               : best_vloss_ft,
                "f1_score"           : val_f1,
            }, CKPT_P2)
            counter = 0
            print(f"  ✓ Saved → {CKPT_P2}")
        else:
            counter += 1
            print(f"  No improvement ({counter}/{PATIENCE})")
            if counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}.")
                break

    writer.close()
    print("\nPhase 2 Done!")

    checkpoint = torch.load(CKPT_P2, map_location=device, weights_only=True)
    model_best.load_state_dict(checkpoint["model_state_dict"])
    model_best.eval()

    print("\n" + "=" * 50)
    print("Final Evaluation — EfficientNet-B0 (Phase 2)")
    print("=" * 50)
    full_eval(train_dl, "Train")
    full_eval(val_dl,   "Val  ")
    full_eval(test_dl,  "Test ")
    print("=" * 50)
