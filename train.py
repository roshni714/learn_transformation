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
DEFAULT_CONFIG = "configs/tnet.yaml"

def get_dataset(dataset_config):
    dataset = dataset_config["name"]
    corruption = dataset_config["corruption"]

    train_transform = transforms.Compose([transforms.ToTensor(),                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]) 
    test_transform = []

    if dataset =="CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True)

        corrupt_testset = CorruptDataset(testset, corruption, dataset)

        input_size = [3, 32, 32]
    elif dataset == "MNIST":
       train_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,),(0.3087,)) ])
       input_size = [1, 28, 28]
       trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
       testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True)

       corrupt_testset = CorruptDataset(testset, corruption, dataset)



    elif dataset == "STL10":

        train_transform = transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]) 
        test_transform = [transforms.Resize(size=224)]

        trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform= train_transform)
        testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transforms.Compose(test_transform))

        corrupt_testset = CorruptDataset(testset, corruption)
        input_size = [3, 224, 224]
    return trainset, corrupt_testset, input_size



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
    train_data, test_data, input_size = get_dataset(config["data_loader"])
    batch_size = config["tnet"]["batch_size"]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)


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
    num_epochs = 60
    initial_params = []
    for tf in transform_list:
        if tf == "rotation":
            initial_params.append(0)
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
    trainer = TransformNetTrainer(transform_net, transform_net_name, transform_list, pretrained_model, train_loader, test_loader, writer_name,  device)

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        trainer.train()
        state = {'state_dict': trainer.transform_net}
        torch.save(state, os.path.join(config["tnet"]["save_dir"], "checkpoint_{}.tar".format(trainer.epoch)))


if __name__ == "__main__":
    main()
