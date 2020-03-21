import torch
import torchvision.transforms as transforms
import numpy as np


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

name_to_range = {"brightness": [0.01, 2],
                 "contrast": [0.01, 2],
                 "hue": [0.01, 2],
                 "saturation": [0.01, 2],
                 "rotation": [-15, 15]}

class Evaluater():

    def __init__(self, model, test_data, device):
        self.model = model
        self.test_data = test_data
        self.device = device

    def evaluate(self):
        print("Starting evaluation")
        transform_list = ["brightness", "contrast", "saturation", "hue", "rotation"]

        for transform in transform_list:
            self.evaluate_over_transform(transform)

        print("Done with evaluation")

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
            for i, sample in enumerate(self.test_data):
                img_batch, label = sample
                for j in range(img_batch.shape[0]):
                    img = img_batch[j]
                    shifted_img = tfs(img).float().to(self.device).unsqueeze(dim=0)

                    pred_class  = torch.argmax(self.model(shifted_img)).detach().cpu()
                    accuracy += torch.sum(label == pred_class)
                    total +=1

            accuracy /= total
            accuracies.append(accuracy)

        print(accuracies)
        return {"x": points, "y": accuracies, "tf": tf} 
                



