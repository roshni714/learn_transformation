import torch
import torchgeometry as tgm

def apply_transform_batch(img_batch, transform_out, transform_list):
        sigmoid = torch.nn.Sigmoid()
        tanh = torch.nn.Tanh()

        transform_act = torch.zeros(1, transform_out.shape[1]).to(img_batch.get_device())
        new_img_batch = torch.zeros(img_batch.shape).to(img_batch.get_device())
        for i in range(img_batch.shape[0]):
            for j, tf in enumerate(transform_list):
                name = tf

                if name == "saturation":
                    sat_act = 2 * sigmoid(transform_out[i][j])
                    transform_act[j] += sat_act
                    new_img_batch[i] = adjust_saturation(img_batch[i], sat_act)
                if name == "brightness":
                    bright_act =0.5 * tanh(transform_out[i][j])
                    transform_act[j] += bright_act
                    new_img_batch[i] = adjust_brightness(img_batch[i], bright_act)
                if name == "rotation":
                    rot_act = 15 * torch.sin(transform_out[i][j])
                    transform_act[j] += rot_act
                    new_img_batch[i] = adjust_rotation(img_batch[i], rot_act)
        transform_act /= new_img_batch.shape[0]
        return new_img_batch, transform_act
            

def adjust_saturation(img, sat):

    device = img.get_device()
    square = torch.pow(img, 2)
    vals = [0.299, 0.587, 0.114]
    mult = torch.ones(img.shape)
    for i in range(len(vals)):
        mult[i, :, :] *= vals[i]
    mult= mult.to(device)
    res = square * mult
    p = res.sum(dim=0).sqrt().unsqueeze(0)

    copy_p = p.repeat(3, 1, 1)

    new_img = copy_p + (img - copy_p) * sat
    return new_img

def adjust_brightness(img, brightness):
    new_img = img + brightness
    return torch.clamp(new_img, 0, 1)

def adjust_rotation(img, rotation):
    img = img.unsqueeze(dim=0)
    angle = torch.ones(1).to(img.get_device()) * rotation
    # define the rotation center
    center = torch.ones(1, 2).to(img.get_device())
    center[..., 0] = img.shape[3] / 2  # x
    center[..., 1] = img.shape[2] / 2  # y
    # define the scale factor
    scale = torch.ones(1).to(img.get_device())

    # compute the transformation matrix
    M = tgm.get_rotation_matrix2d(center, angle, scale)

    # apply the transformation to original image
    _, _, h, w = img.shape
    img_warped = tgm.warp_affine(img, M, dsize=(h, w))
    img_warped = img_warped.squeeze()
    return img_warped
