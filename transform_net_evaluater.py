import torch 
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
import differentiable_transforms as diff_tf
from tensorboardX import SummaryWriter

class TransformNetEvaluater():

    def __init__(self, transform_net, transform_net_name, pretrained_model, original_model_path, oracle_model_path, test_loader, transform_list, device):
        self.transform_net = transform_net
        self.pretrained_model = pretrained_model
        self.transform_net_name = transform_net_name
        self.test_loader = test_loader
        self.device = device
        self.transform_list = transform_list
        self.writer = SummaryWriter("runs/evaluate")
        self.original_model_path = original_model_path
        self.oracle_model_path = oracle_model_path


    def get_performance_with_tnet(self, original_model_path):
        self.load_model(original_model_path)
        self.pretrained_model.eval()

        if self.transform_net_name != "vec":
            self.transform_net.eval()

        pred_correct = []
        for i, data in enumerate(self.test_loader):
            img_batch, label_batch = data
            img_batch_cuda = img_batch.float().to(self.device)
            label_batch = label_batch.to(self.device)

            if self.transform_net_name != "vec":
                transform_params = self.transform_net(img_batch_cuda)
            else:
                transform_params = self.transform_net
            new_img_batch, mean_transform, var_transform  = diff_tf.apply_transform_batch(img_batch_cuda, transform_params, self.transform_list)
            new_img_batch = new_img_batch.to(self.device)
            out = torch.argmax(self.pretrained_model(new_img_batch), dim=1)
            res = list((out==label_batch).type(torch.uint8).cpu().numpy())
            pred_correct.extend(res)

        return np.array(pred_correct)


    def get_performance_with_standard_model(self, path):
        self.load_model(path)
        self.pretrained_model.eval()

        pred_correct = []
        for i, data in enumerate(self.test_loader):
            img_batch, label_batch = data
            img_batch = img_batch.float().to(self.device)
            label_batch = label_batch.to(self.device)
            out = torch.argmax(self.pretrained_model(img_batch), dim=1)
            res = list((out == label_batch).type(torch.uint8).cpu().numpy())
            pred_correct.extend(res)

        return np.array(pred_correct)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.pretrained_model.load_state_dict(checkpoint["state_dict"])
 
    def evaluate(self):
        oracle_pred = self.get_performance_with_standard_model(self.oracle_model_path)
        original_pred  = self.get_performance_with_standard_model(self.original_model_path)
        tnet_pred = self.get_performance_with_tnet(self.original_model_path)

        num_trials = 10
        indices = list(range(len(original_pred)))

        rows = []
        
        for i in range(num_trials):
            sampled = np.random.choice(indices, 1000)
            original_acc = (np.sum(original_pred[sampled])/1000)
            tnet_acc = (np.sum(tnet_pred[sampled])/1000)
            oracle_acc = np.sum(oracle_pred[sampled])/1000

            row = {"tnet_acc": tnet_acc, "pretrained_acc": original_acc , "oracle_acc": oracle_acc}
            rows.append(row)
        return rows
