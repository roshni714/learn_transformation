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
name_to_extra_params = {"rotation": {"resample": 3}}

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

        self.normalized_embeddings, self.trainset_labels, self.mean_embed, self.std_embed = self.generate_trainset_embeddings()
#        self.nearest_neighbors = self.generate_trainset_nearest_neighbors()
        self.centroids = self.generate_trainset_centroids()


    def generate_trainset_embeddings(self):
        #Generate embeddings of original train set
        tfs_original = transforms.Compose([transforms.ToPILImage(), 
                   transforms.ToTensor(), 
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        original_embeddings = []
        labels = []

        self.model.eval()

        for i, data in enumerate(self.train_loader):
            img_batch, label_batch = data
            for j in range(img_batch.shape[0]):
                img = img_batch[j]
                original_img = tfs_original(img).float().to(self.device).unsqueeze(dim=0)
                out_original = self.model.get_layer4(original_img).detach().cpu().numpy().squeeze()
                original_embeddings.append(out_original)
                labels.append(label_batch[j])

        original_embeddings = torch.Tensor(original_embeddings)
        mean = torch.mean(original_embeddings, dim=0)
        std = torch.std(original_embeddings, dim=0)
        normalized_embeddings = (original_embeddings - mean)/std
        return normalized_embeddings, labels, mean, std

    def generate_trainset_centroids(self):
        print("Calculating centroids")

        clusters = [[] for i in range(10)]

        for i, embedding in enumerate(self.normalized_embeddings):
            label = int(self.trainset_labels[i])
            clusters[label].append(embedding.numpy())

        centroids = [torch.Tensor(cluster).mean(dim=0) for cluster in clusters]

        return centroids

    def min_distance_from_centroids(self, vec):
        mse = torch.nn.MSELoss(reduction='sum')

        min_dist = float("inf")
        for center in self.centroids:
            dist = mse(center, vec.squeeze())
            if dist.item() < min_dist:
                min_dist = dist.item()
        return min_dist


    def get_class_of_nearest_centroid(self, vec):
        mse = torch.nn.MSELoss(reduction='sum')

        min_dist = float("inf")
        class_num = None
        for i in range(len(self.centroids)):
            dist = mse(self.centroids[i], vec)
            if dist.item() < min_dist:
                min_dist = dist.item()
                class_num = i
        return class_num


    def generate_trainset_nearest_neighbors(self):
        print("Fit nearest neighbors on training data")
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(self.normalized_embeddings)
        return neigh

    def evaluate(self):
        print("Starting evaluation")
        for transform in self.transform_list:
            self.generate_plots_opt_tf_distribution(transform)



        print("Done with evaluation")


    def embedding_accuracy_over_transform(self, tf):
        print("Evaluating embedding accuracy.")
        min_max = name_to_range[tf]
        points = np.linspace(min_max[0], min_max[1], 15)

        self.model.eval()

        accuracies = []
        embedding_accuracies = []
        for point in points:
            dic = {name_to_arg[tf]: [point - 0.001, point + 0.001]}
            if tf in name_to_extra_params:
                for extra_key in name_to_extra_params[tf]:
                    dic[extra_key] = name_to_extra_params[tf][extra_key]
            tfs = transforms.Compose([transforms.ToPILImage(), 
                   name_to_transform[tf](**dic), 
                   transforms.ToTensor(), 
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            accuracy = 0.
            embedding_accuracy = 0.
            total = 0.
            for i, sample in enumerate(self.test_loader):
                img_batch, label = sample
                for j in range(img_batch.shape[0]):
                    img = img_batch[j]
                    true_label = label[j]
                    shifted_img = tfs(img).float().to(self.device).unsqueeze(dim=0)

                    #Model
                    pred_label = torch.argmax(self.model(shifted_img).detach().cpu()).item()

                    #Embedding
                    out_shifted = self.model.get_layer4(shifted_img).detach().cpu()
                    out_shifted = (out_shifted - self.mean_embed)/self.std_embed

#                    _, train_example_index = self.nearest_neighbors.kneighbors(out_shifted)
                    embedding_label = self.get_class_of_nearest_centroid(out_shifted)

                    if true_label == embedding_label:
                        embedding_accuracy += 1.

                    if true_label == pred_label:
                        accuracy += 1.

                    total += 1.
            accuracy /=total
            embedding_accuracy /=total

            accuracies.append(accuracy)
            embedding_accuracies.append(embedding_accuracy)

        return {"x":list(points), "model_accuracies": accuracies, "embedding_accuracies": embedding_accuracies, "name": tf}


    def get_weighted_distance(self, vec=None, weights=None):
        mse = torch.nn.MSELoss(reduction='sum')
        max_index = torch.argmax(weights[0])

        dist = 0.
        for i in range(len(weights[0])):
            if i != max_index:
                dist += (1/torch.sqrt(torch.sum(torch.pow(self.centroids[i] -vec, 2)))) * weights[0][i]
            else:
                dist += torch.sqrt(torch.sum(torch.pow(self.centroids[i] -vec, 2))) * 1/weights[0][i]
        return dist


    def embedding_distance_over_transform(self, tf):
        print("Calculating embedding distance {}".format(tf))
        #Calculate embedding distance with a sweep over different transformations
        min_max = name_to_range[tf]
        points = np.linspace(min_max[0], min_max[1], 15)

        mean_embedding_distance = []
        std_embedding_distance = []

        self.model.eval()
        mse = torch.nn.MSELoss(reduction='sum')
        softmax = torch.nn.Softmax(dim=1)
        for point in points:
            dic = {name_to_arg[tf]: [point - 0.001, point + 0.001]}
            if tf in name_to_extra_params:
                for extra_key in name_to_extra_params[tf]:
                    dic[extra_key] = name_to_extra_params[tf][extra_key]
            tfs = transforms.Compose([transforms.ToPILImage(), 
                   name_to_transform[tf](**dic), 
                   transforms.ToTensor(), 
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            embedding_error = []
            for i, sample in enumerate(self.test_loader):
                if i > 100:
                    break
                img_batch, label = sample
                for j in range(img_batch.shape[0]):
                    img = img_batch[j]
                    shifted_img = tfs(img).float().to(self.device).unsqueeze(dim=0)

                    last_layer = self.model(shifted_img)
                    pred_class = torch.argmax(last_layer).unsqueeze(dim=0)

                    out_shifted = self.model.get_layer4(shifted_img).detach().cpu()
                    out_shifted = (out_shifted - self.mean_embed)/self.std_embed
                   
                    weights = softmax(last_layer)

                    error = self.get_weighted_distance(out_shifted, weights)
                    embedding_error.append(error)
            embedding_error = torch.Tensor(embedding_error)
            mean_embed_dist = torch.mean(embedding_error)
            std_embed_dist = torch.std(embedding_error)
            mean_embedding_distance.append(mean_embed_dist.item())
            std_embedding_distance.append(std_embed_dist.item())
        return {"x": list(points), "y": mean_embedding_distance, "std": std_embedding_distance, "name": tf}

    def get_opt_tf_distribution(self, tf):
        print("Calculating embedding distance {}".format(tf))
        #Calculate embedding distance with a sweep over different transformations
        min_max = name_to_range[tf]
        points = np.linspace(min_max[0], min_max[1], 15)

        mean_embedding_distance = []
        std_embedding_distance = []

        self.model.eval()
        mse = torch.nn.MSELoss(reduction='sum')
        softmax = torch.nn.Softmax(dim=1)

        optimal_tf_dist= []
        for i, sample in enumerate(self.test_loader):
            if i > 100:
                break
            img_batch, label = sample
            for j in range(img_batch.shape[0]):
                img = img_batch[j]
                softmax_max= []

                for point in points:
                    dic = {name_to_arg[tf]: [point - 0.001, point + 0.001]}
                    if tf in name_to_extra_params:
                        for extra_key in name_to_extra_params[tf]:
                            dic[extra_key] = name_to_extra_params[tf][extra_key]
                    tfs = transforms.Compose([transforms.ToPILImage(), 
                       name_to_transform[tf](**dic), 
                       transforms.ToTensor(), 
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

                    shifted_img = tfs(img).float().to(self.device).unsqueeze(dim=0)
                    last_layer = self.model(shifted_img)
                    weights = softmax(last_layer)

                    softmax_max.append(torch.max(weights))

                index = torch.argmax(torch.Tensor(softmax_max))
                optimal_tf = points[index.item()]
                optimal_tf_dist.append((label[j], optimal_tf)) 

        return {"opt_tf_dist": optimal_tf_dist,  "name": tf}

    def generate_plots_opt_tf_distribution(self, tf):

        dic = self.get_opt_tf_distribution(tf)
        opt_tf_dist, name = dic["opt_tf_dist"], dic["name"].capitalize()

        #Make general histogram
        f = plt.figure()
        ax = f.add_subplot(111)
        all_mins = [opt_tf  for label, opt_tf in opt_tf_dist]
        ax.set_xlabel("{} Parameter".format(name))
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Optimal {} Parameter".format(name))
        ax.hist(all_mins)
        print(name)
        plt.savefig("figs/hist/{}/general.pdf".format(name))
        plt.close()

        f = plt.figure(figsize=(20, 8))

        for i in range(10):
            ax = f.add_subplot(5, 2, i+1)
            class_mins =[ opt_tf for label, opt_tf in  list(filter(lambda x: x[0] !=i, opt_tf_dist))]
            ax.set_xlabel("{} Parameter".format(name))
            ax.set_ylabel("Count")
            ax.set_title("Class {}".format(i))
            ax.hist(class_mins)
        plt.title("Distribution of Optimal {} Parameter".format(name))
        plt.savefig("figs/hist/{}/class_specific.pdf")
        plt.close()

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
            dic = {name_to_arg[tf]: [point - 0.001, point + 0.001]}
            if tf in name_to_extra_params:
                for extra_key in name_to_extra_params[tf]:
                    dic[extra_key] = name_to_extra_params[tf][extra_key]
            tfs = transforms.Compose([transforms.ToPILImage(), 
                   name_to_transform[tf](**dic), 
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
                


