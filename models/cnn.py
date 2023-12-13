import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), stride=1), # 1*28*28 -> 10*26*26
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(2, 2)), # 10*26*26 -> 10*13*13
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), stride=1), # 10*13*13 -> 20*10*10
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(2, 2)), # 20*10*10 -> 20*5*5
            # nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(3, 3), stride=1), # 20*5*5 -> 40*2*2
            # nn.ReLU(), 
            # nn.MaxPool2d(kernel_size=(2, 2)), # 40*2*2 -> 40*1*1
        )
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=20*5*5, out_features=100), # 20*5*5 -> 100
            nn.Linear(in_features=100, out_features=10) # 100 -> 10
        )

    def forward(self, x):
        assert x.dim() > 2
        if x.dim() == 3:
            x = x.view(-1, 1, x.size(-2), x.size(-1))
        x = self.layers(x)
        return self.fc(x.view(-1, 20*5*5))
        # return self.fc(x.view(-1, 40*1*1))
