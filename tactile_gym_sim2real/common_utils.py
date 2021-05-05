import argparse
import os

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def load_gan_model(image_dims):
    """
    Doesn't work yet...
    Load the class containing the correct GAN model based on the image dims
    """
    image_dims = list(image_dims)

    if image_dims == [256,256]:
        components = 'pix2pix.gan_models.models_256_auxrl'.split('.')

    elif image_dims == [128,128]:
        components = 'pix2pix.gan_models.models_128_auxrl'.split('.')

    elif image_dims == [64,64]:
        components = 'pix2pix.gan_models.models_64_auxrl'.split('.')

    else:
        sys.exit('Incorrect dims specified')

    mod = __import__('pybullet_real2sim')

    for comp in components:
        mod = getattr(mod, comp)
        print(mod.__dict__.keys())

    GeneratorUNet = getattr(mod, 'GeneratorUNet')
    Discriminator = getattr(mod, 'Discriminator')

    return GeneratorUNet, Discriminator
