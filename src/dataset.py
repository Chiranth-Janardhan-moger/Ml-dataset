import os
import glob
import random
import torch
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor
from src.config import Config
from src.features import load_and_preprocess_audio, extract_mel_spectrogram_image, extract_sequential_features

class BabyCryDataset(Dataset):
    def __init__(self, file_paths, labels, whisper_feature_extractor=None):
        self.file_paths = file_paths
        self.labels = labels
        if whisper_feature_extractor is None:
            # Load default feature extractor for whisper-small
            self.whisper_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        else:
            self.whisper_extractor = whisper_feature_extractor
            
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # 1. Load and pad/truncate raw audio to 5 seconds
        y, sr = load_and_preprocess_audio(file_path)
        
        # 2. Extract Stream 1: Mel-Spectrogram Image (3, 224, 224)
        visual_features = extract_mel_spectrogram_image(y, sr)
        
        # 3. Extract Stream 2: Handcrafted MFCC + extra features (TIME_STEPS, 125)
        sequential_features = extract_sequential_features(y, sr)
        
        # 4. Extract Stream 3: Whisper feature extractor representation
        # Whisper expects input audio as log-mel spectrogram of shape (80, 3000)
        # We can extract it using huggingface feature extractor
        # Note: Hugging Face feature extractor returns a dictionary, we want the "input_features" key
        whisper_inputs = self.whisper_extractor(y, sampling_rate=sr, return_tensors="pt")
        whisper_features = whisper_inputs.input_features.squeeze(0) # (80, 3000)
        
        return {
            "visual_features": visual_features,      # (3, 224, 224)
            "sequential_features": sequential_features,  # (TIME_STEPS, 125)
            "whisper_features": whisper_features,      # (80, 3000)
            "label": torch.tensor(label, dtype=torch.long),
            "file_path": file_path
        }

def get_train_val_split(data_dir=Config.DATA_DIR, val_split=0.2, seed=42):
    """
    Scans the data directory, maps classes to folder names, and performs a stratified train/val split.
    """
    random.seed(seed)
    
    file_paths = []
    labels = []
    
    # Scan subdirectories
    for folder, label_idx in Config.CLASS_MAP.items():
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            # Try matching other names if folder isn't found exactly
            # (e.g. pain vs belly_pain)
            print(f"Warning: Directory {folder_path} not found.")
            continue
            
        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
        print(f"Found {len(wav_files)} files in class '{folder}' ({Config.LABEL_MAP[label_idx]})")
        
        for f in wav_files:
            file_paths.append(f)
            labels.append(label_idx)
            
    if len(file_paths) == 0:
        raise ValueError(f"No WAV files found in directory {data_dir}. Check dataset paths.")
        
    # Group files by label to perform stratified split
    class_groups = {i: [] for i in Config.CLASS_MAP.values()}
    for fp, label in zip(file_paths, labels):
        class_groups[label].append(fp)
        
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    
    for label, paths in class_groups.items():
        random.shuffle(paths)
        split_idx = int(len(paths) * (1 - val_split))
        
        train_paths.extend(paths[:split_idx])
        train_labels.extend([label] * split_idx)
        
        val_paths.extend(paths[split_idx:])
        val_labels.extend([label] * (len(paths) - split_idx))
        
    # Shuffle train and val sets
    train_indices = list(range(len(train_paths)))
    random.shuffle(train_indices)
    train_paths = [train_paths[i] for i in train_indices]
    train_labels = [train_labels[i] for i in train_indices]
    
    val_indices = list(range(len(val_paths)))
    random.shuffle(val_indices)
    val_paths = [val_paths[i] for i in val_indices]
    val_labels = [val_labels[i] for i in val_indices]
    
    print(f"Total dataset size: {len(file_paths)}")
    print(f"Train split size: {len(train_paths)}")
    print(f"Validation split size: {len(val_paths)}")
    
    return train_paths, train_labels, val_paths, val_labels
