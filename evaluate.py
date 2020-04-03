import argparse
import os
import sys
import torch
import yaml
import modules.network as network
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split
from classification_model_evaluater import Evaluater


DEFAULT_CONFIG = "configs/cifar_eval.yaml"

def get_dataset(dataset_config):
    dataset = dataset_config["name"]
    train_transform = transforms.ToTensor()
    if dataset =="CIFAR10":
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=train_transform)
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
   
    return trainset, testset

def main():
    parser = argparse.ArgumentParser(description="Classifier")
    parser.add_argument('-c', '--config', default=DEFAULT_CONFIG, type=str)
    
    args = parser.parse_args()
    config_file = args.config

    # Load config
    assert os.path.exists(args.config), "Config file {} does not exist".format(args.config)
    
    with open(config_file) as fp:
        config = yaml.load(fp)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))


    #Pretrained Model Architecture

    num_channels = 3
    model_name = "resnet18"
    num_classes = 10
    model = network.get_model(name=model_name,
                              input_size=[3, 32, 32],
                              pretrained=False, 
                              num_channels=num_channels, 
                              num_classes=num_classes)
    model.to(device)
    pretrained_model_path = config["pretrained_model"]["path"]
    checkpoint = torch.load(pretrained_model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)


    train_data, test_data = get_dataset(config["data_loader"])
    batch_size = config["batch_size"]

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    transform_list = config["transform_list"]
    evaluater= Evaluater(model, train_loader, test_loader, transform_list,  device)
    evaluater.evaluate()

if __name__ == "__main__":
    main()
