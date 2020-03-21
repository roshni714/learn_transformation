import argparse
import os
import sys
import torch
import yaml
import modules.network as network
from transform_net_trainer import TransformNetTrainer
import torchvision
import torchvision.transforms as transforms

DEFAULT_CONFIG = "configs/tnet.yaml"

def get_dataset(dataset_config):
    dataset = dataset_config["name"]
    corruption = dataset_config["corruption"]

    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()])
    test_transform = []
    for key in corruption:
        if key == "color_jitter":
            test_transform.append(transforms.ColorJitter(**corruption[key]))
    test_transform.append(transforms.ToTensor())

    if dataset =="CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transforms.Compose(test_transform))
    return trainset, testset

def main():
    parser = argparse.ArgumentParser(description="TransformNet Trainer")
    parser.add_argument('-c', '--config', default=DEFAULT_CONFIG, type=str)
    
    args = parser.parse_args()
    config_file = args.config

    # Load config
    assert os.path.exists(args.config), "Config file {} does not exist".format(args.config)
    
    with open(config_file) as fp:
        config = yaml.load(fp)

    if not os.path.exists(config["tnet"]["save_dir"]):
        os.makedirs(config["tnet"]["save_dir"])

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))


    #Pretrained Model Architecture

    num_channels = 3
    model_name = "resnet18"
    num_classes = 10
    pretrained_model = network.get_model(name=model_name, 
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
    print(num_classes)
    transform_net = network.get_model(name=model_name, 
                              pretrained=True, 
                              num_channels=num_channels, 
                              num_classes=num_classes)
    transform_net.to(device)

    train_data, test_data = get_dataset(config["data_loader"])
    batch_size = config["tnet"]["batch_size"]

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)

    num_epochs = 40


    trainer = TransformNetTrainer(transform_net, transform_list, pretrained_model, train_loader, test_loader, device)

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        trainer.train()
        state = {'state_dict': trainer.transform_net.state_dict()}
        torch.save(state, os.path.join(config["tnet"]["save_dir"], "checkpoint_{}.tar".format(trainer.epoch)))


if __name__ == "__main__":
    main()
