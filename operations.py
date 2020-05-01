import torch
import torchgeometry as tgm
import kornia
from torchvision import transforms

def render_img(img):
    if img.get_device() >= 0:
        new_img = torch.ones(img.shape).to(img.get_device())
    else:
        new_img = torch.ones(img.shape)

    if img.shape[0] == 3:
        mean = (0.4914, 0.4822, 0.4465)
        var = (0.247, 0.243, 0.261)

        for i in range(3):
            new_img[i, :, :] = mean[i] + img[i, :, :] * var[i]
    else:
        mean = (0.1307,)
        var = (0.3087,)

        new_img[0, :, :] = mean[0] + img[0, :, :] * var[0]

    return new_img

def normalize_img(img):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    new_img = normalize(img)
    return new_img

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



def adjust_contrast(img, contrast):
    gray = kornia.rgb_to_grayscale(img)
    if img.get_device() >= 0:
        gray_img = torch.ones(img.shape).to(img.get_device()) * gray.mean()
    else:
        gray_img = torch.ones(img.shape) * gray.mean()
    new_img = gray_img * (1- contrast) + img * contrast
    return new_img


def adjust_hue(img, factor):
    hsv = kornia.rgb_to_hsv(img)
    hsv[0, :, :] += factor
    rgb = kornia.hsv_to_rgb(hsv)
    return rgb


def adjust_brightness(img, factor):
    if img.get_device() >= 0:
        black_img = torch.zeros(img.shape).to(img.get_device())
    else:
        black_img = torch.zeros(img.shape)

    scaled_img = render_img(img)
    tf_img = torch.clamp(black_img * (1 - factor) + scaled_img * factor, 0, 1)
    new_img = normalize_img(tf_img)
    return new_img

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

