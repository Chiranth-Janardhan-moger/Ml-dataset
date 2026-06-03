import numpy as np
import librosa
import torch
import torch.nn.functional as F
from src.config import Config

def load_and_preprocess_audio(file_path, target_sr=Config.SAMPLE_RATE, duration=Config.DURATION):
    """
    Loads and resamples an audio file, padding or truncating it to a fixed duration.
    """
    try:
        # Load audio (automatically resampled to target_sr)
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        # Return silent waveform in case of error
        y = np.zeros(int(target_sr * duration))
        sr = target_sr
        
    target_length = int(target_sr * duration)
    
    # Pad or truncate
    if len(y) > target_length:
        y = y[:target_length]
    else:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
        
    return y, sr

def extract_mel_spectrogram_image(y, sr=Config.SAMPLE_RATE):
    """
    Converts audio waveform into a 128-band Mel-Spectrogram image of shape (3, 224, 224).
    Normalized and repeated to 3 channels for EfficientNet.
    """
    # Extract Mel-Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_fft=Config.N_FFT, 
        hop_length=Config.HOP_LENGTH, 
        n_mels=Config.N_MELS
    )
    
    # Convert to log scale (dB)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1]
    mel_min, mel_max = mel_db.min(), mel_db.max()
    if mel_max - mel_min > 1e-8:
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min)
    else:
        mel_norm = np.zeros_like(mel_db)
        
    # Resize to (224, 224) using PyTorch interpolation
    mel_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    mel_resized = F.interpolate(mel_tensor, size=(224, 224), mode='bilinear', align_corners=False)
    mel_resized = mel_resized.squeeze(0).squeeze(0) # (224, 224)
    
    # Convert to 3 channels (repeat)
    mel_3ch = mel_resized.unsqueeze(0).repeat(3, 1, 1) # (3, 224, 224)
    
    # Normalize with ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    mel_final = (mel_3ch - mean) / std
    
    return mel_final

def extract_sequential_features(y, sr=Config.SAMPLE_RATE):
    """
    Extracts MFCC (40) + Delta + Delta-Delta (120) and 5 extra handcrafted features:
    ZCR, RMS energy, spectral centroid, bandwidth, rolloff.
    Returns array of shape (time_steps, 125).
    """
    # 1. MFCC (40 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=Config.MFCC_COEFFS, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH)
    
    # 2. MFCC Delta and Delta-Delta
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    
    # Concatenate to get 120 features
    mfcc_features = np.concatenate([mfccs, delta_mfccs, delta2_mfccs], axis=0) # (120, T)
    
    # 3. Handcrafted features
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=Config.N_FFT, hop_length=Config.HOP_LENGTH) # (1, T)
    rms = librosa.feature.rms(y=y, frame_length=Config.N_FFT, hop_length=Config.HOP_LENGTH) # (1, T)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH) # (1, T)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH) # (1, T)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH) # (1, T)
    
    # Concatenate all extra features
    extra_features = np.concatenate([zcr, rms, centroid, bandwidth, rolloff], axis=0) # (5, T)
    
    # Combine MFCC and handcrafted features -> shape (125, T)
    combined = np.concatenate([mfcc_features, extra_features], axis=0)
    
    # Transpose to shape (T, 125)
    features_seq = combined.T
    
    # Align to fixed TIME_STEPS
    target_steps = Config.TIME_STEPS
    if len(features_seq) > target_steps:
        features_seq = features_seq[:target_steps]
    else:
        pad_width = ((0, target_steps - len(features_seq)), (0, 0))
        features_seq = np.pad(features_seq, pad_width, mode='constant')
        
    return torch.tensor(features_seq, dtype=torch.float32)
