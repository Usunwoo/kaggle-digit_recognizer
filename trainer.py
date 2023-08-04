import numpy as np
import torch

from copy import deepcopy

class Trainer():

    def __init__(self, model, optimizer, crit, device):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit
        self.device = device

        super().__init__()
    
    def _train(self, x, y, config):
        self.model.train()

        indices = torch.randperm(x.size(0)).to(x.device)
        x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0)
        y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)

        train_loss = 0

        for x_i, y_i in zip(x, y):
            y_pred_i = self.model(x_i)
            loss_i = self.crit(y_pred_i, y_i.squeeze())

            self.optimizer.zero_grad()
            loss_i.backward()

            self.optimizer.step()
            # 메모리 누수 방지를 위해 연결을 끊어줘야함
            train_loss += float(loss_i)

        return train_loss / len(x)

    def _valid(self, x, y, config):
        self.model.eval()

        with torch.no_grad():
            x = x.split(config.batch_size, dim=0)
            y = y.split(config.batch_size, dim=0)

            valid_loss = 0

            for x_i, y_i in zip(x, y):
                y_pred_i = self.model(x_i)
                loss_i = self.crit(y_pred_i, y_i.squeeze())

                valid_loss+=float(loss_i)

        return valid_loss / len(x)

    def train(self, train_loader, valid_loader, config):
        train_history, valid_history = [], []

        best_model = None
        lowest_loss = np.inf

        lowest_epoch = 0
        early_stop = 10

        for i in range(config.n_epochs):
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                train_loss += self._train(x, y, config)
            train_loss /= len(train_loader)

            valid_loss = 0
            for x, y in valid_loader:
                x, y = x.to(self.device), y.to(self.device)
                valid_loss += self._valid(x, y, config)
            valid_loss /= len(valid_loader)

            train_history.append(train_loss)
            valid_history.append(valid_loss)

            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                lowest_epoch = i
                best_model = deepcopy(self.model.state_dict())

            # if (i+1) % config.print_interval == 0:
            #     print(f'epoch {i+1}/{config.n_epochs}: train loss={train_loss:2e}, valid loss={valid_loss:2e}, lowest loss={lowest_loss:2e}')
            print(f'epoch {i+1}/{config.n_epochs}: train loss={train_loss:2e}, valid loss={valid_loss:2e}, lowest loss={lowest_loss:2e}')

            if early_stop > 0 and lowest_epoch + early_stop < i+1:
                print(f'early stop at epoch {i+1}')
                break

        self.model.load_state_dict(best_model)
