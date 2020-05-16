import argparse
import os
import sys
import torch
import yaml
import modules.network as network
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split
from classification_model_trainer import Trainer
from data_loaders import CorruptDataset

DEFAULT_CONFIG = "configs/cifar.yaml"

def get_dataset(dataset_config):
    dataset = dataset_config["name"]
    corruption = dataset_config.get("corruption")
    if dataset =="CIFAR10":
        input_size = [3, 32,32]

        if not corruption:
            train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        else:
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
            trainset = CorruptDataset(trainset, corruption, dataset)

    elif dataset == "MNIST":
       train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,),(0.3087,)) ])
       trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
       input_size = [1, 28, 28]
    elif dataset == "STL10":
       train_transform = transforms.Compose([transforms.Resize(size=224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(224, padding=4),
                                          transforms.ToTensor()])
 
       trainset = torchvision.datasets.STL10(root='./data', split="train", download=True, transform=train_transform)

       input_size = [3, 224, 224]
    train_len = int(0.8* len(trainset))

    train, val = random_split(trainset, [train_len, len(trainset) - train_len])
    return train, val, input_size

def main():
    parser = argparse.ArgumentParser(description="Classifier")
    parser.add_argument('-c', '--config', default=DEFAULT_CONFIG, type=str)
    
    args = parser.parse_args()
    config_file = args.config

    # Load config
    assert os.path.exists(args.config), "Config file {} does not exist".format(args.config)
    
    with open(config_file) as fp:
        config = yaml.load(fp)

    if not os.path.exists(config["save_dir"]):
        os.makedirs(config["save_dir"])

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    train_data, val_data, input_size = get_dataset(config["data_loader"])

    #Pretrained Model Architecture
    if config["data_loader"]["name"] =="MNIST":
        num_channels = 1
    else:
        num_channels = 3
    model_name = "resnet18"
    num_classes = 10
    model = network.get_model(name=model_name,
                              input_size = input_size,
                              pretrained=False, 
                              num_channels=num_channels, 
                              num_classes=num_classes)
    model.to(device)

    batch_size = config["batch_size"]

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)

    num_epochs = 40

    trainer = Trainer(model, train_loader, val_loader, config["save_as"], device)

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        trainer.train()
        state = {'state_dict': trainer.model.state_dict()}
        torch.save(state, os.path.join(config["save_dir"], "checkpoint_{}.tar".format(trainer.epoch)))


if __name__ == "__main__":
    main()
