import argparse
import os
import sys
import torch
import yaml
import modules.network as network
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split
import numpy as np
from classification_model_trainer import Trainer
from data_loaders import CorruptDataset

np.random.seed(0)
DEFAULT_CONFIG = "configs/cifar.yaml"


def get_dataset(dataset_config):
    dataset = dataset_config["name"]
    corruption = dataset_config.get("corruption")
    n_examples = dataset_config["n_examples"]
    shuffled_indices = list(range(10000))
    np.random.shuffle(shuffled_indices)
    tl_indices = shuffled_indices[:7000]

    if dataset =="CIFAR10":
        input_size = [3, 32,32]
        testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
        new_testset = []
        indices = np.random.choice(tl_indices, n_examples)
        for i in indices:
            new_testset.append(testset[i])
        testset = CorruptDataset(new_testset, corruption, dataset)
    elif dataset == "MNIST":
       testset  = torchvision.datasets.MNIST(root='./data', train=False, download=True)
       new_testset = []
       indices = np.random.choice(tl_indices, n_examples)
       for i in indices:
           new_testset.append(testset[i])
       testset = CorruptDataset(new_testset, corruption, dataset)

       input_size = [1, 28, 28]
   
    train_len = int(0.8 * len(testset))

    train, val = random_split(testset, [train_len, len(testset) - train_len])
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

    train, val, input_size = get_dataset(config["data_loader"])

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
    pretrained_model_path = config["pretrained_model"]["path"]
    checkpoint = torch.load(pretrained_model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    batch_size = config["batch_size"]

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)

    num_epochs = 61

    trainer = Trainer(model, train_loader, val_loader, config["save_as"], device, mode="fine_tune")

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        trainer.train()
        if epoch % 10 == 0:
            state = {'state_dict': trainer.model.state_dict()}
            torch.save(state, os.path.join(config["save_dir"], "checkpoint_{}.tar".format(trainer.epoch)))

if __name__ == "__main__":
    main()


