import argparse
import os
import sys
import torch
import yaml
import modules.network as network
from transform_net_evaluater import TransformNetEvaluater
import torchvision
import torchvision.transforms as transforms
import PIL
DEFAULT_CONFIG = "configs/tnet.yaml"

def get_dataset(dataset_config):
    dataset = dataset_config["name"]
    corruption = dataset_config["corruption"]

    train_transform = transforms.Compose([transforms.ToTensor()]) 
    test_transform = []
    for key in corruption:
        if key == "color_jitter":
            test_transform.append(transforms.ColorJitter(**corruption[key]))
        if key == "rotation":
            test_transform.append(transforms.RandomRotation(**corruption[key]))
    test_transform.append(transforms.ToTensor())

    if dataset =="CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transforms.Compose(test_transform))
        input_size = [3, 32, 32]
    elif dataset == "STL10":

        train_transform = transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]) 
        test_transform = [transforms.Resize(size=224)]
        for key in corruption:
            if key == "color_jitter":
                test_transform.append(transforms.ColorJitter(**corruption[key]))
            if key == "rotation":
                test_transform.append(transforms.RandomRotation(**corruption[key]))
        test_transform.append(transforms.ToTensor())

        trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform= train_transform)
        testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transforms.Compose(test_transform))
        input_size = [3, 224, 224]
    return trainset, testset, input_size

def main():
    parser = argparse.ArgumentParser(description="TransformNet Trainer")
    parser.add_argument('-c', '--config', default=DEFAULT_CONFIG, type=str)
    
    args = parser.parse_args()
    config_file = args.config

    # Load config
    assert os.path.exists(args.config), "Config file {} does not exist".format(args.config)
    
    with open(config_file) as fp:
        config = yaml.load(fp)


    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))


    #Load Data and get image size
    _, test_data, input_size = get_dataset(config["data_loader"])
    batch_size = config["batch_size"]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)


    #Pretrained Model Architecture

    num_channels = 3
    model_name = "resnet18"
    num_classes = 10
    pretrained_model = network.get_model(name=model_name,
                              input_size = input_size,
                              pretrained=False, 
                              num_channels=num_channels, 
                              num_classes=num_classes)
    pretrained_model_path = config["pretrained_model"]["path"]
    checkpoint = torch.load(pretrained_model_path)
    pretrained_model.load_state_dict(checkpoint["state_dict"])
    pretrained_model.to(device)

    #Transform Net Architecture
    num_channels = 3
    model_name = "resnet18"
    transform_list = config["tnet"]["transform_list"]
    num_classes = len(transform_list)
    transform_net = network.get_model(name=model_name,
                              input_size=[3, 32, 32],
                              pretrained=True, 
                              num_channels=num_channels, 
                              num_classes=num_classes)
    transform_net.to(device)
    transform_net_path = config["tnet"]["path"]
    checkpoint = torch.load(transform_net_path)
    transform_net.load_state_dict(checkpoint["state_dict"])
    transform_net.to(device)

    num_epochs = 40


    evaluater = TransformNetEvaluater(transform_net, pretrained_model, test_loader, transform_list, device)
    evaluater.evaluate()

if __name__ == "__main__":
    main()
