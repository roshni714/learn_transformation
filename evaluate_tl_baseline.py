import argparse
import os
import sys
import torch
import yaml
import modules.network as network
from tl_evaluater import TLEvaluater
import torchvision
import torchvision.transforms as transforms
import PIL
import csv
from data_loaders import CorruptDataset
import numpy as np
DEFAULT_CONFIG = "configs/tnet.yaml"

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
    pretrained_model.to(device)
    transform_list = config["tl_model"]["transform_list"]
    evaluater = TLEvaluater(pretrained_model, config["tl_model"]["path"], test_loader, transform_list, device)
    rows = evaluater.evaluate()

    tf_name= config["tl_model"]["path"].split("/")[-1]

    
    name = "tl"
    
    with open('{}_{}.csv'.format(name, tf_name), 'a+', newline='') as csvfile:
        field_names = rows[0].keys()
        dict_writer = csv.DictWriter(csvfile, fieldnames=field_names)
        dict_writer.writeheader()
        for row in rows:
            print(row)
            dict_writer.writerow(row)



if __name__ == "__main__":
    main()
