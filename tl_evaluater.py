import torch 
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
import os


class TLEvaluater():

    def __init__(self, empty_model, model_direc, test_loader, transform_list, device):
        self.model = empty_model
        self.test_loader = test_loader
        self.device = device
        self.transform_list = transform_list
        self.model_direc = model_direc

    def get_performance_tl(self):
        self.model.eval()

        pred = []
        for i, data in enumerate(self.test_loader):
            img_batch, label_batch = data
            img_batch = img_batch.float().to(self.device)
            label_batch = label_batch.to(self.device)
            out = torch.argmax(self.model(img_batch), dim=1)
            res = list((out == label_batch).type(torch.uint8).cpu().numpy())
            pred.extend(res)

        return np.array(pred)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["state_dict"])
 
    def evaluate(self):

        num_trials = 10
        rows = []
        
        for config in os.listdir(self.model_direc):
            num = int(config.split(".")[0].split("_")[1])
            path = os.path.join(self.model_direc, config, "checkpoint_61.tar")
            self.load_model(path)
            pred = self.get_performance_tl()

            indices = list(range(len(pred)))
            for i in range(num_trials):
                sampled = np.random.choice(indices, 1000)
                acc = np.sum(pred[sampled]).item()/1000
                row = {"accuracy": acc, "n_examples": num}
                rows.append(row)

        return rows
