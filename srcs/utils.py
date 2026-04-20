import torch
import numpy as np
import matplotlib.pyplot as plt
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score


def train_one_epoch(dataloader, model, loss_fn, optimizer, epoch, device, writer, log_step_interval=50):
    """
    Train model for one epoch.

    Args:
        dataloader        : PyTorch DataLoader
        model             : PyTorch model
        loss_fn           : Loss function
        optimizer         : Optimizer
        epoch             : Current epoch number
        device            : Device (cuda / mps / cpu)
        writer            : TensorBoard SummaryWriter
        log_step_interval : Print + log every N steps
    """
    model.train()
    running_loss = 0.

    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % log_step_interval == 0:
            avg_loss = running_loss / log_step_interval
            print(f"  Epoch {epoch+1}, Step {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}")
            writer.add_scalar("Loss/train_step", loss.item(), epoch * len(dataloader) + i)
            running_loss = 0.


def test(dataloader, model, loss_fn, device):
    """
    Evaluate model on a dataloader.

    Returns:
        loss    : average loss over all batches
        y_preds : predicted class indices (tensor)
        y_trues : ground truth class indices (tensor)
    """
    model.eval()
    y_preds, y_trues = [], []
    total_loss = 0.

    with torch.no_grad():
        for X, y in dataloader:
            X, y   = X.to(device), y.to(device)
            pred   = model(X)
            total_loss += loss_fn(pred, y).item()
            y_preds.append(pred.argmax(1))
            y_trues.append(y)

    return total_loss / len(dataloader), torch.cat(y_preds), torch.cat(y_trues)

def evaluate(dataloader, model, loss_fn, device):
    loss, preds, trues = test(dataloader, model, loss_fn, device)
    acc = multiclass_accuracy(preds, trues).item()
    f1  = multiclass_f1_score(preds, trues).item()
    return loss, acc, f1

def plot_predictions(images, labels, preds, class_names, save_path="sanity_check.jpg"):
    """
    Plot a 3x3 grid of predictions vs ground truth.

    Args:
        images      : batch of images (tensor)
        labels      : ground truth labels (tensor)
        preds       : predicted labels (tensor)
        class_names : list of class name strings
        save_path   : where to save the output image
    """
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        img  = images[i].permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img  = np.clip(std * img + mean, 0, 1)
        ax.imshow(img)

        true_label = class_names[labels[i].item()]
        pred_label = class_names[preds[i].item()]
        color      = "green" if true_label == pred_label else "red"
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Saved prediction plot → '{save_path}'")
