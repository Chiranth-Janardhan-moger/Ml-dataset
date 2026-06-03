import torch
import torch.nn as nn
import torchvision.models as models

class VisualStream(nn.Module):
    """
    Stream 1 — Visual Stream (CNN):
    - Mel-Spectrogram input shape: (batch_size, 3, 224, 224)
    - Uses EfficientNet-B0 pretrained on ImageNet
    - Replaces final layer with 128-dim embedding output
    - Fine-tunes last 3 blocks only
    """
    def __init__(self, embedding_dim=128):
        super(VisualStream, self).__init__()
        
        # Load pretrained EfficientNet-B0
        try:
            # Modern torchvision syntax
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.backbone = models.efficientnet_b0(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            self.backbone = models.efficientnet_b0(pretrained=True)
            
        # Freeze all features first
        for param in self.backbone.features.parameters():
            param.requires_grad = False
            
        # Unfreeze last 3 blocks of EfficientNet-B0 (indices 6, 7, 8 of features)
        # block 6, 7 are MBConv blocks, 8 is the final ConvBNActivation layer
        for i in [6, 7, 8]:
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True
                
        # Replace classification head with a 128-dim embedding projection
        # EfficientNet-B0 classifier input size is 1280
        self.backbone.classifier = nn.Sequential(
            nn.Linear(1280, embedding_dim)
        )
        
    def forward(self, x):
        # Input shape: (batch_size, 3, 224, 224)
        # Output shape: (batch_size, 128)
        return self.backbone(x)
