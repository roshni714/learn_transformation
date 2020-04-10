import torch
from tensorboardX import SummaryWriter
from datetime import datetime
import differentiable_transforms as diff_tf


class TransformNetTrainer():

    def __init__(self, transform_net, transform_list, pretrained_model, train_loader, test_loader, device):
        print("Setting up TransformNetTrainer")

        self.device = device
        self.transform_net = transform_net
        self.pretrained_model = pretrained_model
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.optimizer = torch.optim.Adam(self.transform_net.parameters(), lr=1e-4)

        names ='_'.join(transform_list)

        self.transform_list = transform_list
        timestamp = datetime.timestamp(datetime.now())
        self.writer= SummaryWriter("runs/{}/{}".format(names, timestamp))

        self.epoch = 0
#        self.normalized_embeddings, self.labels, self.mean_embed, self.std_embed = self.generate_train_embeddings()
#        self.centroids = self.generate_train_centroids()

    def generate_train_centroids(self):
        print("Calculating centroids")

        clusters = [[] for i in range(10)]

        for i, embedding in enumerate(self.normalized_embeddings):
            label = int(self.labels[i])
            clusters[label].append(embedding.numpy())

        centroids = [torch.Tensor(cluster).mean(dim=0).to(self.device) for cluster in clusters]

        return centroids

    def generate_train_embeddings(self):
        original_embeddings = []
        labels = []

        self.pretrained_model.eval()

        for i, data in enumerate(self.train_loader):
            img_batch, label_batch = data
            img_batch = img_batch.float().to(self.device)
            out_original = self.pretrained_model.get_layer4(img_batch).detach().cpu().numpy()
            for j in range(out_original.shape[0]):
                original_embeddings.append(out_original[j])
                labels.append(label_batch[j])

        original_embeddings = torch.Tensor(original_embeddings)
        mean = torch.mean(original_embeddings, dim=0)
        std = torch.std(original_embeddings, dim=0)
        normalized_embeddings = (original_embeddings - mean)/std
        return normalized_embeddings, labels, mean, std

    def get_weighted_distance(self, test_batch):

        softmax = torch.nn.Softmax(dim=1)
        vecs = self.pretrained_model.get_layer4(test_batch)
        weights = softmax(self.pretrained_model(test_batch))


        dist = 0.
        for j in range(test_batch.shape[0]):
            max_index = torch.argmax(weights[j])
            for i in range(len(weights[j])):
                if i != max_index:
                    dist += (1/torch.sqrt(torch.sum(torch.pow(self.centroids[i] -vecs[j], 2)))) *weights[j][i]
                else:
                    dist += torch.sqrt(torch.sum(torch.pow(self.centroids[i] -vecs[j], 2))) * (1/weights[j][i])
        return dist


    def msp_loss(self, test_example_batch, transform_param):

        self.pretrained_model.eval()
        
        batch_loss= - torch.min(torch.max(self.pretrained_model(test_example_batch))[0])

        return batch_loss

    def train(self):

        self.pretrained_model.eval()
        self.transform_net.train()


        for i, sample in enumerate(self.test_loader):
            cur_iter = self.epoch*len(self.test_loader) + i

            img, label = sample
            img = img.float().to(self.device)
            
            transform_out  = self.transform_net(img)
            transformed_test_batch, transform_param= diff_tf.apply_transform_batch(img, transform_out, self.transform_list)
            loss = self.msp_loss(transformed_test_batch, transform_param)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar('data/loss', loss.item(), cur_iter)
            for j, tf  in enumerate(self.transform_list):
                self.writer.add_scalar('data/{}'.format(tf), transform_param[j].item(), cur_iter)
                self.writer.add_image('img/corrupted', img[0], cur_iter)
                self.writer.add_image('img/transformed', transformed_test_batch[0], cur_iter)
            print("Epoch: [{0}][{1}/{2}]\t Loss {3}".format(self.epoch, i, len(self.test_loader), round(loss.item(), 4)))
        self.epoch += 1


