import torch
import torch.nn as nn
from transformers import WhisperModel

class PretrainedAudioStream(nn.Module):
    """
    Stream 3 — Pretrained Audio Stream (Whisper):
    - Uses openai/whisper-small encoder as a feature extractor.
    - Input shape: (batch_size, 80, 3000) - log-mel spectrogram features
    - Extracts encoder hidden states
    - Global average pooling -> Dense(128)
    - Freezes whisper weights, training only the projection head
    """
    def __init__(self, embedding_dim=128):
        super(PretrainedAudioStream, self).__init__()
        
        # Load whisper-small pretrained model
        self.whisper = WhisperModel.from_pretrained("openai/whisper-small")
        
        # We only need the encoder to extract audio features (saves memory/compute)
        self.encoder = self.whisper.encoder
        
        # Freeze all whisper encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Whisper-small encoder hidden state dimension is 768
        self.projection = nn.Sequential(
            nn.Linear(768, embedding_dim)
        )
        
    def forward(self, x):
        # x shape: (batch_size, 80, 3000)
        
        # Run Whisper Encoder
        # The output of encoder is a BaseModelOutput (we want last_hidden_state)
        encoder_outputs = self.encoder(x)
        hidden_states = encoder_outputs.last_hidden_state  # Shape: (batch_size, seq_len, 768)
        
        # Global Average Pooling over the time/sequence dimension (dim 1)
        # Shape: (batch_size, 768)
        pooled = hidden_states.mean(dim=1)
        
        # Project to 128-dimensional embedding
        # Shape: (batch_size, 128)
        embedding = self.projection(pooled)
        
        return embedding
