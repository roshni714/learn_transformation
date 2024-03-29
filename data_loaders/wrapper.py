from torch.utils.data import Dataset
import data_loaders.corrupt_transforms as corrupt_tf
import numpy as np
import torchvision.transforms as transforms


class CorruptDataset(Dataset):

    def __init__(self, dataset, corruption, name):
        self.dataset = dataset
        rs = np.random.RandomState(0)
        self.name = name 
        self.params = {}
        n = len(self.dataset)
        self.mode = corruption.get("mode", "test")
        if "mode" in corruption:
            del corruption["mode"]

        for key in corruption:
            if not key.endswith("_var"):
               self.params[key] = rs.normal(corruption[key], corruption.get("{}_var".format(key), 0.), n)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tf_dic = {}
        for key in self.params:
            tf_dic[key] = self.params[key][idx]

        img, label  = self.dataset[idx]

        if self.mode == "test":
            if self.name != "MNIST":
                tf = transforms.Compose([corrupt_tf.Corruption(**tf_dic), transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            else:
                tf = transforms.Compose([corrupt_tf.Corruption(**tf_dic), transforms.ToTensor(),
                                         transforms.Normalize((0.1307,),(0.3087,))]) 
        else:
            if self.name != "MNIST":
                tf = transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.RandomCrop(32, padding=4),
                                         corrupt_tf.Corruption(**tf_dic), transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
            else:
                tf = transforms.Compose([corrupt_tf.Corruption(**tf_dic), transforms.ToTensor(),
                                         transforms.Normalize((0.1307,),(0.3087,))]) 
 

        return tf(img), label



def WrapperDataset(Dataset):

    def __init__(self, imgs, labels):
        self.imgs = img
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]


def get_few_shot_dataset(dataset, num_examples_per_class):

    classes = set([label for img, label in self.dataset])

    imgs = []
    labels = []

    classes_count = {i: 0 for i in classes}

    for img, label in dataset:
        if label in classes:
            classes_count[label] += 1
            imgs.append(img)
            labels.append(label)
            if classes_count[label] == num_examples_per_class:
                classes.remove(label)
        if len(classes) == 0:
            break

    dataset = WrapperDataset(imgs, labels)

    print("Size of Few Shot Dataset:{}".format( len(dataset)))
    return dataset
    


    
