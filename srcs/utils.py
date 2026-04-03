import torch
import numpy as np
import matplotlib.pyplot as plt

def train_one_epoch(
        dataloader, model, 
        loss_fn, 
        optimizer, epoch, 
        device, writer,
        log_step_interval=50):
    """
    Train a multi-task model for one epoch.

    Args:
        dataloader: PyTorch DataLoader
        model: PyTorch model
        loss_fn: Loss function (MAE)
        optimizer: Optimizer (Adam, SGD, etc.)
        epoch: Current epoch number
        device: Device to train on (CPU/GPU)
        writer: TensorBoard writer
        log_step_interval: Logging interval
    """
    model.train()

    running_loss = 0.
    for i, (X, y) in enumerate(dataloader):
        X, y    = X.to(device), y.to(device)

        optimizer.zero_grad()

        # Model prediction
        preds   = model(X)

        loss    = loss_fn(preds, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i+1) % log_step_interval == 0:
            print(f"Epoch {epoch+1}, Step {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
            writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + i)

def test(dataloader, model, loss_fn, device):
    num_batches         = len(dataloader)
    y_preds, y_trues    = [], []
    loss                = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y    = X.to(device), y.to(device)
            
            pred    = model(X)
            y_pred  = pred.argmax(1)

            loss    += loss_fn(pred, y).item()

            y_preds.append(y_pred)
            y_trues.append(y)
    y_preds     = torch.cat(y_preds)
    y_trues     = torch.cat(y_trues)

    loss /= num_batches
    return loss, y_preds, y_trues

def plot_predictions(images, labels, preds, class_names):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    
    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1) # Ensure pixel values are strictly valid
        
        ax.imshow(img)
        
        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        text_color = "green" if true_label == pred_label else "red"
        
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=text_color)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig('sanity_check.jpg')
    print("Saved prediction plot to 'sanity_check.jpg'")