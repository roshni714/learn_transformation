import torch
import operations as op
from torchvision import transforms

tf_to_method = {"saturation": op.adjust_saturation, 
                "brightness": op.adjust_brightness,
                "rotation": op.adjust_rotation,
                "hue": op.adjust_hue,
                "contrast": op.adjust_contrast}

def apply_transform_batch(img_batch, transform_out, transform_list):
        sigmoid = torch.nn.Sigmoid()
        tanh = torch.nn.Tanh()
        softplus = torch.nn.Softplus()

        mean_transform = torch.zeros(1, transform_out.shape[1]).to(img_batch.get_device())
        variance_transform = torch.zeros(1, transform_out.shape[1]).to(img_batch.get_device())

#        new_img_batch = torch.zeros(img_batch.shape).to(img_batch.get_device())
        new_img_batch = img_batch
        for j, tf in enumerate(transform_list):
            name = tf
            if name == "contrast":
                act = softplus(transform_out[:, j])
            if name == "saturation":
                act = sigmoid(transform_out[:, j])
            if name == "brightness":
                act = softplus(transform_out[:, j])
            if name == "rotation":
                act = 180* torch.sin(transform_out[:, j])
            if name == "hue":
                act = 0.5*tanh(transform_out[:, j])

            mean_transform[:, j] += torch.mean(act)

            if act.shape[0] == 1:
                variance_transform[:, j] = 0
            else:
                variance_transform[:, j] += torch.var(act)
            new_img_batch = adjust_tf_batch(new_img_batch, act, tf_to_method[name])




#        new_img_batch = img_batch
#        for j, tf in enumerate(transform_list):
#            name = tf
#            new_img_batch = adjust_tf_batch(new_img_batch, transform_act[j], tf_to_method[name])

        return new_img_batch, mean_transform, variance_transform
            

def apply_transform_batch_nondiff(img_batch, transform_out, transform_list):
        sigmoid = torch.nn.Sigmoid()
        tanh = torch.nn.Tanh()
        softplus = torch.nn.Softplus()

        transform_act = torch.zeros(1, transform_out.shape[1])
        new_img_batch = torch.zeros(img_batch.shape)
        for j, tf in enumerate(transform_list):
            name = tf
            if name == "saturation":
                sat_act = sigmoid(transform_out[:, j])
                transform_act[j] += torch.mean(sat_act)
            if name == "brightness":
                bright_act = softplus(transform_out[:, j])
                transform_act[j] += torch.mean(bright_act)
            if name == "rotation":
                rot_act = 180 * torch.sin(transform_out[:, j])
                transform_act[j] += torch.mean(rot_act)
            if name == "hue":
                hue_act = 0.5 * tanh(transform_out[:, j])
                transform_act[j] += hue_act
        
        new_img_batch = img_batch
        for j, tf in enumerate(transform_list):
            name = tf

            if name == "saturation":
                new_img_batch = adjust_tf_batch(new_img_batch, transform_act[j], tf_to_method[name])
            if name == "brightness":
                new_img_batch = adjust_tf_batch(new_img_batch, transform_act[j], tf_to_method[name])
            if name == "rotation":
                new_img_batch = adjust_rotation_nondiff(new_img_batch, transform_act[j])
            if name == "hue":
                new_img_batch = adjust_tf_btach(new_img_batch, transform_act[j], tf_to_method[name])
         
        return new_img_batch, transform_act

def adjust_rotation_nondiff(img_batch, rot):

    new_img_batch = torch.zeros(img_batch.shape)

    for i in range(img_batch.shape[0]):
        img = img_batch[i]
        tfs = transforms.Compose([transforms.ToPILImage(), transforms.Resize((35, 35), interpolation=0),
                   transforms.RandomRotation(degrees=[rot.item() -0.001, rot.item()+0.001], resample=3),
                   transforms.CenterCrop((32, 32)),
                   transforms.ToTensor()]) 
        new_img_batch[i]= tfs(img).float().unsqueeze(dim=0)

    return new_img_batch

def adjust_brightness_nondiff(img_batch, brightness):

    new_img_batch = torch.zeros(img_batch.shape)

    for i in range(img_batch.shape[0]):
        img = img_batch[i]
        tfs = transforms.Compose([transforms.ToPILImage(), transforms.ColorJitter(brightness=brightness.item()),
                                  transforms.ToTensor()]) 
        new_img_batch[i]= tfs(img).float().unsqueeze(dim=0)

    return new_img_batch


def adjust_tf_batch(img_batch, act, method):

    if img_batch.get_device() >= 0:
        new_img_batch = torch.zeros(img_batch.shape).to(img_batch.get_device())
    else:
        new_img_batch = torch.zeros(img_batch.shape)

    if act.shape[0] == img_batch.shape[0]:
        index = True
    else:
        index = False
    for i in range(img_batch.shape[0]):
        if index:
            new_img_batch[i] = method(img_batch[i], act[i])
        else:
            new_img_batch[i] = method(img_batch[i], act)
    return new_img_batch



