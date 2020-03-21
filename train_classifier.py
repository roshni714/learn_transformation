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

DEFAULT_CONFIG = "configs/cifar.yaml"

def get_dataset(dataset_config):
    dataset = dataset_config["name"]
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if dataset =="CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    
    train_len = int(0.8* len(trainset))

    train, val = random_split(trainset, [train_len, len(trainset) - train_len])
    return train, val

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


    #Pretrained Model Architecture

    num_channels = 3
    model_name = "resnet18"
    num_classes = 10
    model = network.get_model(name=model_name, 
                              pretrained=False, 
                              num_channels=num_channels, 
                              num_classes=num_classes)
    model.to(device)

    train_data, val_data = get_dataset(config["data_loader"])
    batch_size = config["batch_size"]

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)

    num_epochs = 40

    trainer = Trainer(model, train_loader, val_loader, device)

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        trainer.train()
        state = {'state_dict': trainer.model.state_dict()}
        torch.save(state, os.path.join(config["save_dir"], "checkpoint_{}.tar".format(trainer.epoch)))


if __name__ == "__main__":
    main()
