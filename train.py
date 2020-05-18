import argparse
import os
import sys
import torch
import yaml
import modules.network as network
from transform_net_trainer import TransformNetTrainer
import torchvision
import torchvision.transforms as transforms
import PIL
from data_loaders import CorruptDataset, get_few_shot_dataset
from temperature_scaling import ModelWithTemperature
import numpy as np
DEFAULT_CONFIG = "configs/tnet.yaml"

np.random.seed(0)

def get_dataset(dataset_config):
    dataset = dataset_config["name"]
    corruption = dataset_config["corruption"]
    shuffled_indices = list(range(10000))
    np.random.shuffle(shuffled_indices)
    tnet_indices = shuffled_indices[:7000]
    print(tnet_indices[:10])

    if dataset =="CIFAR10":
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True)
        new_testset = []
        for i in tnet_indices:
            new_testset.append(testset[i])

        corrupt_testset = CorruptDataset(new_testset, corruption, dataset)

        input_size = [3, 32, 32]
    elif dataset == "MNIST":
       input_size = [1, 28, 28]
       testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True)

       new_testset = []
       for i in tnet_indices:
           new_testset.append(testset[i])

       corrupt_testset = CorruptDataset(new_testset, corruption, dataset)


    return corrupt_testset, input_size



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


    #Load Data and get image size
    test_data, input_size = get_dataset(config["data_loader"])
    batch_size = config["tnet"]["batch_size"]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)


    #Pretrained Model Architecture
    if config["data_loader"]["name"] == "MNIST":
        num_channels = 1
    else:
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
    transform_list = config["tnet"]["transform_list"]
    num_classes = len(transform_list)
    num_epochs = config["tnet"].get("epochs") or 60
    initial_params = []
    for tf in transform_list:
        if tf == "rotation":
            initial_params.append(0)
            initial_params.append(1)
        else:
            initial_params.append(1)

    if config["tnet"].get("name") == "vec":
        transform_net_name = "vec"

        transform_net = torch.Tensor([initial_params]).to(device).requires_grad_(True)
    elif config["tnet"]["name"] == "resnet18":
        transform_net_name = "resnet18"
        transform_net = network.get_model(name="resnet18",
                              input_size=[3, 32, 32],
                              pretrained=False, 
                              num_channels=num_channels, 
                              num_classes=num_classes).to(device)

        weight_dict = transform_net.state_dict()
        new_weight_dict = {"linear.bias": torch.Tensor(initial_params)}
        weight_dict.update(new_weight_dict)
        transform_net.load_state_dict(weight_dict)
    else:
        transform_net_name = "cnn"
        transform_net = network.get_model(name="cnn",
                              input_size=[3, 32, 32],
                              pretrained=False, 
                              num_channels=num_channels, 
                              num_classes=num_classes).to(device)

        weight_dict = transform_net.state_dict()
        new_weight_dict = {"fc3.bias": torch.Tensor(initial_params)}
        weight_dict.update(new_weight_dict)
        transform_net.load_state_dict(weight_dict)

    writer_name = config["tnet"]["save_as"]
    trainer = TransformNetTrainer(transform_net, transform_net_name, transform_list, pretrained_model, test_loader, writer_name,  device)

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        trainer.train()
        state = {'state_dict': trainer.transform_net}
        torch.save(state, os.path.join(config["tnet"]["save_dir"], "checkpoint_{}.tar".format(trainer.epoch)))


if __name__ == "__main__":
    main()
