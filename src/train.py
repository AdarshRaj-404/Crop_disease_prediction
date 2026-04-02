import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import get_model
from tqdm import tqdm

def train_model(num_epochs=5, learning_rate=0.001, model_save_path='best_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    train_loader, val_loader, _, classes = get_dataloaders()
    num_classes = len(classes)
    
    # 2. Initialize Model
    model = get_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    # 3. Define Loss & Optimizer - Added weight decay (L2 Regularization) for overfitting
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    early_stop_patience = 5
    epochs_no_improve = 0
    
    # 4. Training Loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)
        
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)
            
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")
        
        # Save best model and Early Stopping
        if val_epoch_loss < best_val_loss:
            print(f"Validation loss decreased ({best_val_loss:.4f} --> {val_epoch_loss:.4f}). Saving model...")
            loss_file = os.path.dirname(model_save_path)
            if loss_file and not os.path.exists(loss_file):
                os.makedirs(loss_file)
            torch.save(model.state_dict(), model_save_path)
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epochs.")
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered. Stopping training.")
                break

    print("Training complete.")

if __name__ == '__main__':
    # You can change epochs, lr, etc here.
    save_path = os.path.join(os.path.dirname(__file__), '..', 'best_model.pth')
    train_model(num_epochs=3, learning_rate=0.001, model_save_path=save_path)
