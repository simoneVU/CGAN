from CGAN.generator import Generator
from dataset.fashionMNIST import load_F_MNIST,imsave

import torch 

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=100, help="dim. of latent space")
parser.add_argument("--img_dim", type=int, default=28, help="size of both image dimensions")
parser.add_argument("--num_classes", type=int, default=10, help="nr. of classes")
args = parser.parse_args()

latent_dim = args.latent_dim
n_classes = args.num_classes
image_shape = [args.img_dim, args.img_dim]

dataset = load_F_MNIST()
model = Generator(latent_dim, n_classes, image_shape)
model.load_state_dict(torch.load('saved_models/generator_weights.pth'))
model.eval()

for label in np.arange(1,10):
    for i in range(len(load_F_MNIST())):
        if dataset[i][1] == label:
            true_image, true_label = dataset[i][0], dataset[i][1]
            sample_image = model(torch.randn(1, latent_dim), torch.tensor([true_label]))
            imsave([sample_image, true_image], ["Generated Image", "True Image"], label)
            break