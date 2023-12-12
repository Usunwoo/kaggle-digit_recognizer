import torch.nn as nn


class ConvolutionBlock(nn.Module):

    def __init__(self, 
                 input_channels, 
                 output_channels):

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, (3, 3), padding=1), 
            nn.ReLU(), 
            nn.BatchNorm2d(output_channels), 
            nn.Conv2d(output_channels, output_channels, (3, 3), stride=2, padding=1), 
            nn.ReLU(), 
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x):        
        return self.layers(x)
    

class CNNClassifier(nn.Module):
    
    def __init__(self, output_size):
        
        super().__init__()

        self.blocks = nn.Sequential(
            ConvolutionBlock(1, 32),
            ConvolutionBlock(32, 64),
            ConvolutionBlock(64, 128),
            ConvolutionBlock(128, 256),
            ConvolutionBlock(256, 512),
        )

        self.layers = nn.Sequential(
            nn.Linear(512, 256), 
            nn.ReLU(), 
            nn.BatchNorm1d(256), 
            nn.Linear(256, 128), 
            nn.ReLU(), 
            nn.BatchNorm1d(128), 
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.BatchNorm1d(64), 
            nn.Linear(64, output_size), 
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        assert x.dim() > 2

        if x.dim() == 3:
            x = x.view(-1, 1, x.size(-2), x.size(-1))
            
        z = self.blocks(x)
        return self.layers(z.squeeze())