import os
from datetime import datetime
import kagglehub
import shutil
import torch
from torch import nn
from torchvision import datasets
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_f1_score
)
from utils import train_one_epoch, test, plot_predictions
from model import ResNet50

learning_rate = 5e-4
batch_size = 64
epochs = 15

train_trans = transforms_v2.Compose([
    transforms_v2.Resize((224, 224)),
    transforms_v2.RandomHorizontalFlip(p=0.5),
    transforms_v2.RandomRotation(degrees=5), # Very slight rotation
    transforms_v2.ColorJitter(brightness=0.1, contrast=0.1),
    transforms_v2.ToImage(),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_trans = transforms_v2.Compose([
    transforms_v2.Resize((224, 224)),
    transforms_v2.CenterCrop(224),
    transforms_v2.ToImage(), 
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
])

if __name__ == '__main__':

    data_dir = r"./UTKFace_organized"

    base_train_dataset = datasets.ImageFolder(root=data_dir, transform=train_trans)
    base_test_dataset = datasets.ImageFolder(root=data_dir, transform=test_trans)


    total_size = len(base_train_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices, test_indices = random_split(
        range(total_size), [train_size, val_size, test_size], generator=generator
    )

    train_ds = Subset(base_train_dataset, train_indices)
    val_ds = Subset(base_test_dataset, val_indices)
    test_ds = Subset(base_test_dataset, test_indices)


    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device\n")
    print("If cuda not found, try installing torch and torchvision with the following command:")
    print("pip uninstall torch torchvision")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130")
        
    # Import weight from resnet50 pretrained on ImageNet

    # Create a model
    model, weights = ResNet50(num_classes=4, freeze_backbone=True)
    model = model.to(device)

    print(f"Model successfully loaded and sent to: {device}")

    # Try to feed `batch_x` into the model to test the forward pass
    batch_x, batch_y = next(iter(train_dl))
    y_hat = model(batch_x.to(device))
    y_pred = torch.argmax(y_hat, dim=1)

    plot_predictions(batch_x, batch_y, y_pred, test_ds.dataset.classes)

    ####################
    # Model Training
    ####################

    # Setup tensorboard
    writer = SummaryWriter(f'./runs/trainer_{model._get_name()}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    # Specify loss function
    class_weights = torch.tensor([1.89, 1.0, 1.45, 1.88]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # Specify optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    best_vloss = 100000.
    if not os.path.exists('model_best_vloss.pth'):
    # if True:
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} / {epochs}")
            train_one_epoch(train_dl, model, loss_fn, optimizer, epoch, device, writer, log_step_interval=1)
            train_loss, train_y_preds, train_y_trues = test(train_dl, model, loss_fn, device)
            val_loss, val_y_preds, val_y_trues = test(val_dl, model, loss_fn, device)
            
            # Performance metrics
            train_perf = {
                'accuracy': multiclass_accuracy(train_y_preds, train_y_trues).item(),
                'f1': multiclass_f1_score(train_y_preds, train_y_trues).item(),
            }
            
            # Performance metrics
            val_perf = {
                'accuracy': multiclass_accuracy(val_y_preds, val_y_trues).item(),
                'f1': multiclass_f1_score(val_y_preds, val_y_trues).item(),
            }
            
            # Log model training performance
            writer.add_scalars('Train vs. Valid/loss', 
                {'train':train_loss, 'valid': val_loss}, 
                epoch)
            writer.add_scalars(
                'Performance/acc', 
                {'train':train_perf['accuracy'], 'valid': val_perf['accuracy']},
                epoch)
            writer.add_scalars(
                'Performance/f1', 
                {'train':train_perf['f1'], 'valid': val_perf['f1']},
                epoch)

            # Track best performance, and save the model's state
            if val_loss < best_vloss:
                best_vloss = val_loss
                torch.save(model.state_dict(), 'model_best_vloss.pth')
                print('Saved best model to model_best_vloss.pth')
        print("Done!")


    ###########################
    # Evaluate on the Test Set
    ###########################
 

    model_best, _ = ResNet50(num_classes=len(test_ds.dataset.classes), freeze_backbone=False)
    model_best = model_best.to(device)
    checkpoint = torch.load('model_best_vloss.pth', map_location=device, weights_only=True)
    model_best.load_state_dict(checkpoint)

    model_best.eval()


    train_loss, train_y_preds, train_y_trues = test(train_dl, model_best, loss_fn, device)

    # Performance metrics on the training set
    train_perf = {
        'accuracy': multiclass_accuracy(train_y_preds, train_y_trues).item(),
        'f1': multiclass_f1_score(train_y_preds, train_y_trues).item(),
    }

    # Use the best model on the test set
    test_loss, test_y_preds, test_y_trues = test(test_dl, model_best, loss_fn, device)

    # Performance metrics
    test_perf = {
        'accuracy': multiclass_accuracy(test_y_preds, test_y_trues).item(),
        'f1': multiclass_f1_score(test_y_preds, test_y_trues).item(),
    }

    print(f"Train: loss={train_loss:>8f}, acc={(100*train_perf['accuracy']):>0.1f}%, f1={(100*train_perf['f1']):>0.1f}%")
    print(f"Test: loss={test_loss:>8f}, acc={(100*test_perf['accuracy']):>0.1f}%, f1={(100*test_perf['f1']):>0.1f}%")


    #########################################
    # PHASE 2: Fine-Tuning Layer 4
    #########################################

    print("\n--- Starting Phase 2: Fine-Tuning Layer 4 ---")
    model_best.train()
    patience = 15
    counter = 0
    # Unfreeze all layers of your loaded best model
    for param in model_best.parameters():
        param.requires_grad = True

    optimizer_ft = torch.optim.AdamW([
        {'params': model_best.conv1.parameters(), 'lr': 1e-7},
        {'params': model_best.layer1.parameters(), 'lr': 1e-7},
        {'params': model_best.layer2.parameters(), 'lr': 1e-6},
        {'params': model_best.layer3.parameters(), 'lr': 1e-6},
        {'params': model_best.layer4.parameters(), 'lr': 1e-5},
        {'params': model_best.fc.parameters(), 'lr': 1e-4}
    ], weight_decay=0.05)

    epochs_ft = 60
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=epochs_ft)
    best_vloss_ft = 100000.
    # if not os.path.exists('model_best_finetuned.pth'):
    if True:
        for epoch in range(epochs_ft):
            print(f"Fine-Tune Epoch {epoch+1} / {epochs_ft} (LR: {optimizer_ft.param_groups[-1]['lr']:.6f})")
            

            train_one_epoch(train_dl, model_best, loss_fn, optimizer_ft, epoch, device, writer, log_step_interval=1)
            scheduler.step()

            train_loss, train_y_preds, train_y_trues = test(train_dl, model_best, loss_fn, device)
            val_loss, val_y_preds, val_y_trues = test(val_dl, model_best, loss_fn, device)
            
            if val_loss < best_vloss_ft:
                best_vloss_ft = val_loss
                torch.save(model_best.state_dict(), 'model_best_finetuned.pth')
                counter = 0
                print('Saved best FINE-TUNED model to model_best_finetuned.pth')
            else:
                counter += 1
                print(f"No improvement in validation loss for {counter} epochs.")
                if counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break

    print("\nPhase 2 Done!")

    # Final Evaluation of the Fine-Tuned Model
    model_best.load_state_dict(torch.load('model_best_finetuned.pth', map_location=device, weights_only=True))
    model_best.eval()

    train_loss, train_y_preds, train_y_trues = test(train_dl, model_best, loss_fn, device)

    # Performance metrics on the training set
    train_perf = {
        'accuracy': multiclass_accuracy(train_y_preds, train_y_trues).item(),
        'f1': multiclass_f1_score(train_y_preds, train_y_trues).item(),
    }

    # Use the best model on the test set
    test_loss, test_y_preds, test_y_trues = test(test_dl, model_best, loss_fn, device)

    # Performance metrics
    test_perf = {
        'accuracy': multiclass_accuracy(test_y_preds, test_y_trues).item(),
        'f1': multiclass_f1_score(test_y_preds, test_y_trues).item(),
    }
    # Use the best model on the validation set
    val_loss, val_y_preds, val_y_trues = test(val_dl, model_best, loss_fn, device)

    # Performance metrics
    val_perf = {
        'accuracy': multiclass_accuracy(val_y_preds, val_y_trues).item(),
        'f1': multiclass_f1_score(val_y_preds, val_y_trues).item(),
    }

    print(f"Train: loss={train_loss:>8f}, acc={(100*train_perf['accuracy']):>0.1f}%, f1={(100*train_perf['f1']):>0.1f}%")
    print(f"Test: loss={test_loss:>8f}, acc={(100*test_perf['accuracy']):>0.1f}%, f1={(100*test_perf['f1']):>0.1f}%")
    print(f"Validation: loss={val_loss:>8f}, acc={(100*val_perf['accuracy']):>0.1f}%, f1={(100*val_perf['f1']):>0.1f}%")
    
