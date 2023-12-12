from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

class Trainer():
    def __init__(self, model, loss_fn, optimizer, device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def _train(self, train_loader):
        self.model.train()

        train_loss = 0
        for x, y in tqdm(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            train_loss+=loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return train_loss / len(train_loader)

    def _valid(self, valid_loader):
        self.model.eval()
        
        with torch.no_grad():
            valid_loss, correct = 0, 0
            total = 0
            for x, y in valid_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                valid_loss+=loss

                _, predicted = torch.max(pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        accuracy = correct / total
        return valid_loss / len(valid_loader), accuracy

    def train(self, train_loader, valid_loader, epochs):
        best_epoch, best_model = 0, None
        lowest_loss = np.inf
        for i in range(epochs):
            print(f'=== Epoch {i+1} / {epochs} ===')
            train_loss = self._train(train_loader)
            valid_loss, accuracy = self._valid(valid_loader)
            print(f'train_loss: {train_loss:.7f}, valid_loss: {valid_loss:.7f}, accuracy: {accuracy * 100:.4f}%')
            
            if valid_loss < lowest_loss:
                lowest_loss = valid_loss
                best_epoch = i+1
                best_model = deepcopy(self.model.state_dict())

        print(f"Finish. best epoch: {best_epoch}, lowest loss: {lowest_loss}")
        self.model.load_state_dict(best_model)
