import torchvision.transforms.functional as TF

class Corruption:

    def __init__(self, degrees=0., brightness=1., contrast=1., saturation=1., hue=0.):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.degrees = degrees

    def __call__(self, x):
        if self.degrees != 0.:
            x = TF.rotate(x, self.degrees, resample=3)
        if self.brightness != 1.:
            x = TF.adjust_brightness(x, self.brightness)
        if self.contrast != 1.:
            x = TF.adjust_contrast(x, self.contrast)
        if self.hue != 0.:
            x = TF.adjust_hue(x, self.hue)
        if self.saturation != 1.:
            x = TF.adjust_saturation(x, self.saturation)
        return x

