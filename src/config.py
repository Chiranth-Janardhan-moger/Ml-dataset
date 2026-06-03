import os
import torch

class Config:
    # Path settings
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "cry-dataset")
    SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
    
    # Audio parameters
    SAMPLE_RATE = 16000  # 16 kHz for Whisper and consistency
    DURATION = 5.0       # Truncate/pad to 5 seconds
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    
    # Feature configurations
    TIME_STEPS = 156     # 5.0 * 16000 / 512 = 156.25 -> 156 frames
    MFCC_COEFFS = 40
    SEQUENTIAL_INPUT_DIM = 125 # 40 MFCC + 40 Delta + 40 Delta-Delta + 5 extra features
    
    # Model parameters
    NUM_CLASSES = 5
    EMBEDDING_DIM = 128
    FUSION_DIM = 384     # 128 * 3
    
    # Training hyperparameters
    BATCH_SIZE = 8       # Small batch size since audio and pretrained models are heavy
    LEARNING_RATE = 1e-4
    EPOCHS = 15
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Class maps (Directory name to label index)
    CLASS_MAP = {
        "hungry": 0,
        "belly_pain": 1,
        "tired": 2,
        "discomfort": 3,
        "burping": 4
    }
    
    # Label to readable mapping
    LABEL_MAP = {
        0: "Hunger",
        1: "Pain (Belly Pain)",
        2: "Sleepy (Tired)",
        3: "Discomfort",
        4: "Burping"
    }

# Create directories if they do not exist
os.makedirs(Config.SAVE_DIR, exist_ok=True)
