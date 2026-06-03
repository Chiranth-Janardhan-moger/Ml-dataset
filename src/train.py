import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.config import Config
from src.dataset import BabyCryDataset, get_train_val_split
from src.models.fusion_model import SilentCryDecoder
import numpy as np

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move data to device
        visual_input = batch["visual_features"].to(device)
        seq_input = batch["sequential_features"].to(device)
        whisper_input = batch["whisper_features"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(visual_input, seq_input, whisper_input)
        loss = criterion(logits, labels)
        
        # Backward pass & optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item() * visual_input.size(0)
        _, preds = torch.max(logits, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}")
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            visual_input = batch["visual_features"].to(device)
            seq_input = batch["sequential_features"].to(device)
            whisper_input = batch["whisper_features"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(visual_input, seq_input, whisper_input)
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * visual_input.size(0)
            _, preds = torch.max(logits, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def run_training():
    print("="*60)
    print("        TRAINING THE SILENT CRY DECODER MODEL")
    print("="*60)
    
    # 1. Get Train/Val splits
    try:
        train_paths, train_labels, val_paths, val_labels = get_train_val_split()
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        return
        
    # 2. Initialize Datasets and Dataloaders
    print("Initializing datasets...")
    train_dataset = BabyCryDataset(train_paths, train_labels)
    val_dataset = BabyCryDataset(val_paths, val_labels)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=0  # Use 0 workers to avoid Windows multiprocessing issues
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )
    
    # 3. Build Model
    print(f"Building 3-stream fusion model on device: {Config.DEVICE}")
    model = SilentCryDecoder().to(Config.DEVICE)
    
    # 4. Loss, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    # Fine-tune with a small learning rate
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    # 5. Training loop
    best_val_acc = 0.0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    
    print("\nStarting Training...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        print("-" * 20)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, Config.DEVICE)
        
        scheduler.step()
        
        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        elapsed = time.time() - start_time
        print(f"Epoch Summary: Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"               Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        print(f"               Time elapsed: {elapsed:.2f} seconds")
        
        # Checkpoint saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(Config.SAVE_DIR, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"New best model saved to {checkpoint_path} with Val Acc: {val_acc*100:.2f}%")
            
    # Save final model state
    final_checkpoint_path = os.path.join(Config.SAVE_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")
    
    # Save history to JSON
    history_path = os.path.join(Config.SAVE_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    print("\nTraining completed successfully!")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    
if __name__ == "__main__":
    run_training()
