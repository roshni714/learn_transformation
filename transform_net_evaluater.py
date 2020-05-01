import torch 
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
import differentiable_transforms as diff_tf
from tensorboardX import SummaryWriter

class TransformNetEvaluater():

    def __init__(self, transform_net, transform_net_name, pretrained_model, test_loader, transform_list, device):
        self.transform_net = transform_net
        self.pretrained_model = pretrained_model
        self.transform_net_name = transform_net_name
        self.test_loader = test_loader
        self.device = device
        self.transform_list = transform_list
        self.writer = SummaryWriter("runs/evaluate") 


    def get_performance_with_tnet(self):
        self.pretrained_model.eval()

        if self.transform_net_name != "vec":
            self.transform_net.eval()

        total = 0
        correct = 0
        mean_tf = []
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
            mean_tf.append(mean_transform.detach().cpu().numpy())
            out = torch.argmax(self.pretrained_model(new_img_batch), dim=1)

            correct += torch.sum(out == label_batch).item()
            total += img_batch.shape[0]
            self.writer.add_image('img/transformed', new_img_batch[0], i)

        accuracy = correct/total
        print(mean_tf)
        mean_tf = torch.mean(torch.Tensor(mean_tf), dim=0)
        print(mean_tf)
        dic = {}
        for j, tf in enumerate(self.transform_list):
            dic["{}_correction_param".format(tf)] = mean_tf[j].item()
            dic["{}_var".format(tf)] = torch.mean(var_transform[:, j], dim=0).item()

        dic["tnet_acc"] = accuracy
        return dic


    def get_performance_with_pretrained_model(self):
        self.pretrained_model.eval()

        total = 0
        correct = 0
        for i, data in enumerate(self.test_loader):
            img_batch, label_batch = data
            img_batch = img_batch.float().to(self.device)
            label_batch = label_batch.to(self.device)
            out = torch.argmax(self.pretrained_model(img_batch), dim=1)
            correct += torch.sum(out == label_batch).item()
            total += img_batch.shape[0]

        accuracy = correct/total
        return accuracy

    def evaluate(self):
        acc_pretrained = self.get_performance_with_pretrained_model()
        dic  = self.get_performance_with_tnet()
        dic["pretrained_acc"] = acc_pretrained
        return dic
