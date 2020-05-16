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
import csv
from data_loaders import CorruptDataset
DEFAULT_CONFIG = "configs/tnet.yaml"
import numpy as np

np.random.seed(0)
keys = {"brightness": "brightness", "contrast": "contrast", "rotation":"degrees"}

def get_dataset(dataset_config):
    dataset = dataset_config["name"]
    corruption = dataset_config["corruption"]
    shuffled_indices = list(range(10000))
    np.random.shuffle(shuffled_indices)
    tnet_indices = shuffled_indices[7000:]

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


    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))


    #Load Data and get image size
    test_data, input_size = get_dataset(config["data_loader"])
    batch_size = config["batch_size"]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


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
    original_model_path = config["pretrained_model"]["path"]
    oracle_model_path = config["oracle_model"]["path"]
    pretrained_model.to(device)

    #Transform Net Architecture
    name = config["tnet"].get("name") 
    transform_list = config["tnet"]["transform_list"]
    num_classes = len(transform_list)
    transform_net_path = config["tnet"]["path"]

    if name == "vec":
        transform_net_name = "vec"
        transform_net= torch.load(transform_net_path)["state_dict"].to(device)
    else:
        num_channels = 3
        transform_net_name = "resnet18"
        transform_net = network.get_model(name=transform_net_name,
                              input_size=[3, 32, 32],
                              pretrained=True, 
                              num_channels=num_channels, 
                              num_classes=num_classes)
        transform_net= torch.load(transform_net_path)["state_dict"].to(device)


    evaluater = TransformNetEvaluater(transform_net, transform_net_name, pretrained_model, original_model_path, oracle_model_path, test_loader, transform_list, device)
    rows = evaluater.evaluate()

    for row in rows:
        for i, tf in enumerate(transform_list):
            row["{}_param".format(tf)] = config["data_loader"]["corruption"][keys[tf]]

    tf_name= "_".join(transform_list)

    if transform_net_name != "vec":
        name = "results_dist_"
    else:
        name = "results"
    
    with open('{}_{}.csv'.format(name, tf_name), 'a+', newline='') as csvfile:
        field_names = rows[0].keys()
        dict_writer = csv.DictWriter(csvfile, fieldnames=field_names)
        if os.path.getsize('{}_{}.csv'.format(name, tf_name)) == 0:
            dict_writer.writeheader()
        for row in rows:
            dict_writer.writerow(row)



if __name__ == "__main__":
    main()
