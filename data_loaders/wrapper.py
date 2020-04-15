from torch.utils.data import Dataset
import data_loaders.corrupt_transforms as corrupt_tf
import numpy as np
import torchvision.transforms as transforms

class CorruptDataset(Dataset):

    def __init__(self, dataset, corruption):
        self.dataset = dataset

        self.params = {}
        n = len(self.dataset)
        for key in corruption:
            if not key.endswith("_var"):
               self.params[key] = np.random.normal(corruption[key], corruption.get("{}_var".format(key), 0.), n)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tf_dic = {}
        for key in self.params:
            tf_dic[key] = self.params[key][idx]

        tf = transforms.Compose([corrupt_tf.Corruption(**tf_dic), transforms.ToTensor()])
        img, label  = self.dataset[idx]

        return tf(img), label



