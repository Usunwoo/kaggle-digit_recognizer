import torch
import torch.nn as nn

# num_layers 미구현
class VanillaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=10, batch_first=False, device = None):
        super(VanillaLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.device = device
        
        # Forget gate 파라미터
        self.Wxf = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Whf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bf = nn.Parameter(torch.Tensor(hidden_size))

        # Input gate 파라미터
        self.Wxi = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Whi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bi = nn.Parameter(torch.Tensor(hidden_size))

        # Candidate Cell state 파라미터
        self.Wxc = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Whc = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bc = nn.Parameter(torch.Tensor(hidden_size))

        # Output gate 파라미터
        self.Wxo = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Who = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bo = nn.Parameter(torch.Tensor(hidden_size))
        
        self.params = nn.ParameterList([self.Wxf, self.Whf, self.bf, 
                                        self.Wxi, self.Whi, self.bi, 
                                        self.Wxc, self.Whc, self.bc, 
                                        self.Wxo, self.Who, self.bo]).to(device)
        
        # 파라미터에 Xavier 초기화 적용(기존은 0 Tensor)
        self.init_weights(self.params)
        
        self.fc = nn.Linear(hidden_size, num_classes).to(device)

    def init_weights(self, params):
        for param in params:
            if param.data.ndimension() >= 2:
                nn.init.xavier_uniform_(param.data)
            else:
                nn.init.zeros_(param.data)

    def forward(self, x):
        # batch_first가 True이면 sequence_length * batch_size * hidden_size 로 변경해서 계산
        if self.batch_first:
            x = x.permute(1, 0, 2)
        h_prev = torch.zeros((x.size(1), self.hidden_size), device=self.device)
        c_prev = torch.zeros((x.size(1), self.hidden_size), device=self.device)

        h_states = []
        # c_states = []

        for t in range(x.size(0)):
            xt = x[t, :, :] # (batch_size, hidden_size)
            # Forget gate
            ft = torch.sigmoid(torch.matmul(xt, self.Wxf) + torch.matmul(h_prev, self.Whf) + self.bf)
            # Input gate
            it = torch.sigmoid(torch.matmul(xt, self.Wxi) + torch.matmul(h_prev, self.Whi) + self.bi)
            # Output gate
            ot = torch.sigmoid(torch.matmul(xt, self.Wxo) + torch.matmul(h_prev, self.Who) + self.bo)
            # Candidate Cell state (Cell state 후보)
            cct = torch.tanh(torch.matmul(xt, self.Wxc) + torch.matmul(h_prev, self.Whc) + self.bc)
            # Cell state 업데이트
            ct = ft * c_prev + it * cct
            # Hidden state 업데이트
            ht = ot * torch.tanh(ct)

            h_states.append(ht)
            # c_states.append(ct)

            h_prev = ht
            c_prev = ct

        h_states = torch.stack(h_states, dim=0).to(self.device)
        # c_states = torch.stack(c_states, dim=0)

        output = h_states.permute(1, 0, 2) if self.batch_first else h_states
        # FC layer 추가
        output = output[:, -1, :]
        output = self.fc(output)

        # LSTM 원래 return = output, (ht, ct)
        return output


class LSTM(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_layers=1, num_classes=10):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        output, _ = self.lstm(x, (h_0, c_0)) # output = batch_size * sequence_length * hidden_size
        # output, _ = self.lstm(x) # output = batch_size * sequence_length * hidden_size
        output = output[:, -1, :] # 맨 마지막 sequance의 hidden_size만 가져옴
        output = self.fc(output)
        return output
