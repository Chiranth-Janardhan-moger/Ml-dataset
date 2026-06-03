import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """
    Standard scaled dot-product Self-Attention layer that processes sequences 
    of shape (batch_size, seq_len, d_model) and returns (batch_size, seq_len, d_model)
    preserving the sequence dimension.
    """
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5
        
    def forward(self, x):
        # x: (B, T, d_model)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Calculate attention scores
        # scores: (B, T, T)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # output: (B, T, d_model)
        output = torch.matmul(attn_weights, V)
        return output

class SequentialStream(nn.Module):
    """
    Stream 2 — Sequential Stream (BiLSTM):
    - Input shape: (batch_size, time_steps, 125)
    - BiLSTM(128, return_sequences=True) -> Attention layer -> BiLSTM(64)
    - Output: 128-dim embedding
    """
    def __init__(self, input_dim=125, lstm1_hidden=128, lstm2_hidden=64):
        super(SequentialStream, self).__init__()
        
        # First BiLSTM: input_dim -> lstm1_hidden (bidirectional)
        # Outputs shape (B, T, lstm1_hidden * 2) = (B, T, 256)
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm1_hidden,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention layer: d_model = 256
        self.attention = SelfAttention(d_model=lstm1_hidden * 2)
        
        # Second BiLSTM: lstm1_hidden * 2 -> lstm2_hidden (bidirectional)
        # Outputs shape (B, T, lstm2_hidden * 2) = (B, T, 128)
        self.lstm2 = nn.LSTM(
            input_size=lstm1_hidden * 2,
            hidden_size=lstm2_hidden,
            batch_first=True,
            bidirectional=True
        )
        
    def forward(self, x):
        # Input x shape: (B, T, 125)
        
        # Pass through first BiLSTM
        # out1 shape: (B, T, 256)
        out1, _ = self.lstm1(x)
        
        # Pass through sequence-preserving Self-Attention
        # attn_out shape: (B, T, 256)
        attn_out = self.attention(out1)
        
        # Pass through second BiLSTM
        # out2 shape: (B, T, 128)
        # h_n shape: (num_layers * num_directions, B, hidden_size) = (2, B, 64)
        out2, (h_n, _) = self.lstm2(attn_out)
        
        # Concatenate final hidden states from forward and backward passes
        # h_n[0] is forward, h_n[1] is backward for single layer
        # Output shape: (B, 128)
        embedding = torch.cat((h_n[0], h_n[1]), dim=1)
        
        return embedding
