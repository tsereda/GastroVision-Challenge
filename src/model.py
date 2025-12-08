import torch.nn as nn
import timm


class SwinClassifier(nn.Module):
    """Swin Transformer for 4-class endoscopic image classification"""
    
    def __init__(
        self, 
        model_name='swin_base_patch4_window12_384',
        num_classes=4,
        pretrained=True,
        dropout=0.3,
        stochastic_depth=0.2
    ):
        super().__init__()
        
        # Load Swin backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            drop_rate=dropout,
            drop_path_rate=stochastic_depth
        )
        
        # Custom classifier head
        num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
