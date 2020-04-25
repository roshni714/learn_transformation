import torch
import torchvision.models.vgg as models

class AutoencoderLoss():

    def __init__(self):
        self.mse = torch.nn.MSELoss("sum")
        self.vgg = models.vgg_16(pretrained=True)



    def __call__(self, input_batch, target_batch, pred_img_batch, pred_perc_batch):

        reconstruction_error = self.mse(predicted_batch, target_batch)

        perceptual_error = 


