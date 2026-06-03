import torch
import torch.nn as nn
from src.models.visual_stream import VisualStream
from src.models.sequential_stream import SequentialStream
from src.models.pretrained_stream import PretrainedAudioStream
from src.config import Config

class SilentCryDecoder(nn.Module):
    """
    The Silent Cry Decoder - 3-Stream Feature Fusion Model:
    1. Visual Stream (CNN / EfficientNet-B0) -> 128-dim embedding
    2. Sequential Stream (BiLSTM + Self Attention) -> 128-dim embedding
    3. Pretrained Audio Stream (Whisper-small Encoder) -> 128-dim embedding
    
    Fusion Layer:
    - Concatenate streams -> 384-dim vector
    - Dense(256, ReLU) -> BatchNorm -> Dropout(0.4)
    - Dense(128, ReLU) -> Dropout(0.3)
    - Dense(5) -> Raw logits (softmax applied during inference)
    """
    def __init__(self, num_classes=Config.NUM_CLASSES, embedding_dim=Config.EMBEDDING_DIM):
        super(SilentCryDecoder, self).__init__()
        
        # Define the three input streams
        self.visual_stream = VisualStream(embedding_dim=embedding_dim)
        self.sequential_stream = SequentialStream() # Outputs 128-dim embedding automatically
        self.pretrained_stream = PretrainedAudioStream(embedding_dim=embedding_dim)
        
        # Fusion Classifier Head
        self.fc_block1 = nn.Sequential(
            nn.Linear(Config.FUSION_DIM, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.4)
        )
        
        self.fc_block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        
        # Output layer (5 classes)
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, visual_input, sequential_input, whisper_input):
        """
        Args:
            visual_input: Mel-Spectrogram tensors of shape (B, 3, 224, 224)
            sequential_input: Handcrafted sequence features of shape (B, TIME_STEPS, 125)
            whisper_input: Whisper spectrogram features of shape (B, 80, 3000)
        Returns:
            logits: Unnormalized classification logits of shape (B, 5)
        """
        # 1. Forward pass through individual streams
        # Each outputs shape: (B, 128)
        visual_emb = self.visual_stream(visual_input)
        seq_emb = self.sequential_stream(sequential_input)
        whisper_emb = self.pretrained_stream(whisper_input)
        
        # 2. Concat embeddings along dimension 1
        # Combined shape: (B, 384)
        fused = torch.cat((visual_emb, seq_emb, whisper_emb), dim=1)
        
        # 3. Classifier blocks
        x = self.fc_block1(fused)
        x = self.fc_block2(x)
        logits = self.classifier(x)
        
        return logits
        
    def predict(self, visual_input, sequential_input, whisper_input):
        """
        Runs prediction returning class probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(visual_input, sequential_input, whisper_input)
            probs = torch.softmax(logits, dim=-1)
        return probs
