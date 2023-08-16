import argparse

import torch
from torch import nn
from torch import optim

from classification.data_loader import get_loaders
from classification.models.cnn import CNNClassifier
from classification.models.rnn import RNNClassifier
from classification.models.lstm import LSTMClassifier
from classification.trainer import Trainer


def define_argparser():
    p = argparse.ArgumentParser()

    # p.add_argument('--model_fn', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    
    p.add_argument('--train_ratio', type=float, default=0.8)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--early_stop', type=int, default=10)
    p.add_argument('--act', type=str, default='ReLU')
    
    p.add_argument('--hidden_size', type=int, default=64)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--dropout_p', type=float, default=.2)

    config = p.parse_args()

    return config


def main(config):
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device(f'cuda:{config.gpu_id}')

    train_loader, valid_loader = get_loaders(config)
    
    if config.model == "cnn":
        model = CNNClassifier(10).to(device)
    elif config.model == "rnn":
        model = RNNClassifier(
            input_size=28,
            hidden_size=config.hidden_size, 
            output_size=10,
            n_layers=config.n_layers,
            dropout_p=config.dropout_p
        ).to(device)
    elif config.model == "lstm":
        model = LSTMClassifier(
            input_size=28,
            hidden_size=config.hidden_size, 
            output_size=10,
            n_layers=config.n_layers,
            dropout_p=config.dropout_p
        ).to(device)

    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    trainer = Trainer(model, optimizer, crit, device)
    trainer.train(train_loader, valid_loader, config)

    torch.save({'model' : trainer.model.state_dict(), 'config' : config}
                , f"{config.model}.pth")

if __name__ == '__main__':
    config = define_argparser()
    main(config)
