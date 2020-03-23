import torch
import torchvision.transforms as transforms
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import NearestNeighbors


name_to_transform = {"brightness": transforms.ColorJitter,
                     "contrast": transforms.ColorJitter,
                     "hue": transforms.ColorJitter,
                     "saturation": transforms.ColorJitter,
                     "rotation": transforms.RandomRotation}

name_to_arg = {"brightness": "brightness",
               "contrast": "contrast",
               "hue": "hue",
               "saturation": "saturation",
               "rotation": "degrees"}

name_to_range = {"brightness": [0.20, 2],
                 "contrast": [0.20, 2],
                 "hue": [-0.499, 0.499],
                 "saturation": [0.20, 2],
                 "rotation": [-15, 15]}

name_to_extreme = {"brightness": 0.7,
                   "contrast": 0.7,
                   "hue": -0.3,
                   "saturation": 0.7,
                   "rotation": 10}
class Evaluater():

    def __init__(self, model, train_loader, test_loader, transform_list, device):
        self.model = model
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.device = device
        self.transform_list = transform_list

        self.nearest_neighbors, self.mean_embed, self.std_embed = self.generate_trainset_nearest_neighbors()

    def generate_trainset_nearest_neighbors(self):
        #Generate embeddings of original train set
        tfs_original = transforms.Compose([transforms.ToPILImage(), 
                   transforms.ToTensor(), 
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        original_embeddings = []

        self.model.eval()

        for i, data in enumerate(self.train_loader):
            img_batch, _ = data
            for j in range(img_batch.shape[0]):
                img = img_batch[j]
                original_img = tfs_original(img).float().to(self.device).unsqueeze(dim=0)
                out_original = self.model(original_img).detach().cpu().numpy().squeeze()
                original_embeddings.append(out_original)

        original_embeddings = torch.Tensor(original_embeddings)
        mean = torch.mean(original_embeddings, dim=0)
        std = torch.std(original_embeddings, dim=0)
        normalized_embeddings = (original_embeddings - mean)/std
        print("Fit nearest neighbors on training data")
        neigh = NearestNeighbors(n_neighbors=10)
        neigh.fit(normalized_embeddings)
        return neigh, mean, std

    def evaluate(self):
        print("Starting evaluation")

#        for transform in self.transform_list:
#            res = self.evaluate_over_transform(transform)
#            print(res)

        for transform in self.transform_list:
            res = self.embedding_distance_over_transform(transform)
            print(res)
        print("Done with evaluation")

    def generate_tsne_plot_over_transform(self, tf=None):
        def get_color_map(labels):
            NUM_COLORS = 15
            cm = plt.get_cmap('gist_rainbow')
            colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
            color_map = [colors[i] for i in labels]
            return color_map
        
        if not tf:
            name = "original"
            tfs = transforms.Compose([transforms.ToPILImage(), 
                   transforms.ToTensor(), 
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        else:
            name = tf
            point = name_to_extreme[tf]
            key = {name_to_arg[tf]: [point - 0.001, point + 0.001]}
            tfs = transforms.Compose([transforms.ToPILImage(), 
                   name_to_transform[tf](**key), 
                   transforms.ToTensor(), 
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        print("Generating TSNE plot {}".format(name))

        labels = []
        embeddings = []
        for i, data in enumerate(self.test_loader):
            if i > 100:
                break
            else:
                img_batch, label = data
                for j in range(img_batch.shape[0]):
                    img = img_batch[j]
                    shifted_img = tfs(img).float().to(self.device).unsqueeze(dim=0)
                    out = self.model(shifted_img).detach().cpu().numpy().squeeze()
                    labels.append(label[j])
                    embeddings.append(out)

        tsne = TSNE(n_components=2, random_state=0, perplexity=10, n_iter=5000, learning_rate=200)
        print("Fitting tsne...")
        X_2d = tsne.fit_transform(embeddings)
        plt.figure(figsize=(6, 5))
        color_map = get_color_map(labels)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=color_map)
        plt.title("{} CIFAR-10 TSNE Visualization".format(name.capitalize()))
        plt.savefig("figs/tsne_{}.pdf".format("cifar_{}".format(name)))
        plt.close()


    def embedding_distance_over_transform(self, tf):
        print("Calculating embedding distance {}".format(tf))
        #Calculate embedding distance with a sweep over different transformations
        min_max = name_to_range[tf]
        points = np.linspace(min_max[0], min_max[1], 15)

        mean_embedding_distance = []
        std_embedding_distance = []

        mse = torch.nn.MSELoss(reduction='sum')
        for point in points:
            key = {name_to_arg[tf]: [point - 0.001, point + 0.001]}
            tfs = transforms.Compose([transforms.ToPILImage(), 
                   name_to_transform[tf](**key), 
                   transforms.ToTensor(), 
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            embedding_error = []
            for i, sample in enumerate(self.train_loader):
                if i > 100:
                    break
                img_batch, label = sample
                for j in range(img_batch.shape[0]):
                    img = img_batch[j]
                    shifted_img = tfs(img).float().to(self.device).unsqueeze(dim=0)
                    out_shifted = self.model(shifted_img).detach().cpu()
                    out_shifted = (out_shifted - self.mean_embed)/self.std_embed
                    min_error, _ = self.nearest_neighbors.kneighbors(out_shifted)
                    embedding_error.append(np.mean(min_error))
            embedding_error = torch.Tensor(embedding_error)
            mean_embed_dist = torch.mean(embedding_error)
            std_embed_dist = torch.std(embedding_error)
            mean_embedding_distance.append(mean_embed_dist.item())
            std_embedding_distance.append(std_embed_dist.item())
        return {"x": points, "y": mean_embedding_distance, "std": std_embedding_distance, "name": tf}


    def generate_tsne_comparison_plot(self, tf):
        def get_color_map(labels):
            NUM_COLORS = 3
            cm = plt.get_cmap('gist_rainbow')
            colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
            color_map = [colors[i] for i in labels]
            return color_map
        
        tfs_original = transforms.Compose([transforms.ToPILImage(), 
                   transforms.ToTensor(), 
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        point = name_to_extreme[tf]
        key = {name_to_arg[tf]: [point - 0.001, point + 0.001]}
        tfs = transforms.Compose([transforms.ToPILImage(), 
                   name_to_transform[tf](**key), 
                   transforms.ToTensor(), 
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        print("Generating TSNE comparsion plot {}".format(tf))

        labels = []
        embeddings = []
        self.model.eval()
        for i, data in enumerate(self.test_loader):
            if i > 50:
                break
            else:
                img_batch, _ = data
                for j in range(img_batch.shape[0]):
                    img = img_batch[j]
                    shifted_img = tfs(img).float().to(self.device).unsqueeze(dim=0)
                    original_img = tfs_original(img).float().to(self.device).unsqueeze(dim=0)

                    out_shifted = self.model(shifted_img).detach().cpu().numpy().squeeze()
                    out_original = self.model(original_img).detach().cpu().numpy().squeeze()

                    labels.append(0)
                    labels.append(1)
                    embeddings.append(out_original)
                    embeddings.append(out_shifted)

        tsne = TSNE(n_components=2, random_state=0, perplexity=10, n_iter=5000, learning_rate=200)
        print("Fitting tsne...")
        print(len(labels), len(embeddings))
        X_2d = tsne.fit_transform(embeddings)
        plt.figure(figsize=(6, 5))
        color_map = get_color_map(labels)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=color_map)
        plt.title("Original vs. {} CIFAR-10 TSNE Visualization".format(tf.capitalize()))
        plt.savefig("figs/tsne_comparison_{}.pdf".format("cifar_{}".format(tf)))
        plt.close()


    def evaluate_over_transform(self, tf):
        print("Evaluating over {}".format(tf))

        self.model.eval()
        min_max = name_to_range[tf]
        points = np.linspace(min_max[0], min_max[1], 15)

        accuracies = []
        for point in points:
            key = {name_to_arg[tf]: [point - 0.001, point + 0.001]}
            tfs = transforms.Compose([transforms.ToPILImage(), 
                   name_to_transform[tf](**key), 
                   transforms.ToTensor(), 
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            accuracy = 0.
            total = 0.
            for i, sample in enumerate(self.test_loader):
                img_batch, label = sample
                for j in range(img_batch.shape[0]):
                    img = img_batch[j]
                    shifted_img = tfs(img).float().to(self.device).unsqueeze(dim=0)

                    pred_class  = torch.argmax(self.model(shifted_img)).detach().cpu()
                    accuracy += torch.sum(label[j] == pred_class).item()
                    total +=1.

            accuracy /= total
            print(accuracy)
            accuracies.append(accuracy)

        return {"x": points, "y": accuracies, "tf": tf} 
                



