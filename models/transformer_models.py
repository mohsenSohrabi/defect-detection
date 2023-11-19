import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers)
        
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        if x.dim() == 3 and x.size(1) != x.size(2):
            x = x.transpose(0, 1)
        x = self.encoder(x)
        x = x.mean(dim=0)  # Take the mean of the sequence elements
        x = self.classifier(x)
        return x

