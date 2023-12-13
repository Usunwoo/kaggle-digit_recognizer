import torch
from torch import nn

import data_loader
from models.dnn import DNN
from models.cnn import CNN
from trainer import Trainer

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"device: {device}")

    batch_size = 64
    # learning_rate = 0.01
    epochs = 20

    train_loader, valid_loader = data_loader.get_loaders(batch_size, custom_data=True)

    # model = DNN().to(device)
    model = CNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    trainer = Trainer(model, loss_fn, optimizer, device)
    trainer.train(train_loader, valid_loader, epochs)

    # torch.save(trainer.model, f"./model_files/model_DNN.pth")
    torch.save(trainer.model, f"./model_files/model_CNN.pth")

if __name__ == '__main__':
    main()
