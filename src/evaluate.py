import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from src.config import Config
from src.dataset import BabyCryDataset, get_train_val_split
from src.models.fusion_model import SilentCryDecoder

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Standard normalization of CM
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Label inside cells
    fmt = '.2f'
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm_norm[i, j], fmt) + f"\n({cm[i, j]})",
                     horizontalalignment="center",
                     color="white" if cm_norm[i, j] > thresh else "black")
                     
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix plot saved to {save_path}")

def run_evaluation():
    print("="*60)
    print("        EVALUATING THE SILENT CRY DECODER")
    print("="*60)
    
    # 1. Get Val split
    try:
        _, _, val_paths, val_labels = get_train_val_split()
    except Exception as e:
        print(f"Error preparing validation dataset: {e}")
        return
        
    # 2. Loader
    val_dataset = BabyCryDataset(val_paths, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 3. Initialize Model and Load Best Weights
    model = SilentCryDecoder().to(Config.DEVICE)
    checkpoint_path = os.path.join(Config.SAVE_DIR, "best_model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Train the model first.")
        return
        
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 4. Predict
    all_preds = []
    all_labels = []
    
    print("Running predictions on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            visual_input = batch["visual_features"].to(Config.DEVICE)
            seq_input = batch["sequential_features"].to(Config.DEVICE)
            whisper_input = batch["whisper_features"].to(Config.DEVICE)
            labels = batch["label"].to(Config.DEVICE)
            
            logits = model(visual_input, seq_input, whisper_input)
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 5. Calculate Metrics
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Validation Accuracy: {acc*100:.2f}%\n")
    
    # Classification report
    target_names = [Config.LABEL_MAP[i] for i in range(Config.NUM_CLASSES)]
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("Classification Report:")
    print("-" * 60)
    print(report)
    print("-" * 60)
    
    # Save text report
    report_path = os.path.join(Config.SAVE_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Overall Validation Accuracy: {acc*100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Text report saved to {report_path}")
    
    # 6. Plot & Save Confusion Matrix
    cm_path = os.path.join(Config.SAVE_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, target_names, cm_path)

if __name__ == "__main__":
    run_evaluation()
