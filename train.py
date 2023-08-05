import argparse

import torch
from torch import nn
from torch import optim

from trainer import Trainer
from model import ConvolutionalClassifier
from data_loader import get_loaders


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    
    p.add_argument('--train_ratio', type=float, default=0.8)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=256)

    config = p.parse_args()

    return config


def main(config):
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device(f'cuda:{config.gpu_id}')

    train_loader, valid_loader = get_loaders(config)
    
    model = ConvolutionalClassifier(10).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    trainer = Trainer(model, optimizer, crit, device)
    trainer.train(train_loader, valid_loader, config)

    torch.save({'model' : trainer.model.state_dict(), 'config' : config}
                , config.model_fn)

if __name__ == '__main__':
    config = define_argparser()
    main(config)
