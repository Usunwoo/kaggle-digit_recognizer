import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_layers=1, num_classes=10):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        output, _ = self.rnn(x, h_0) # output = batch_size * sequence_length * hidden_size
        output = output[:, -1, :] # 맨 마지막 sequance의 hidden_size만 가져옴
        output = self.fc(output)
        return output
