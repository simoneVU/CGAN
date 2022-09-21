from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt

def load_F_MNIST(train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))])
    data = datasets.FashionMNIST('~/.pytorch/F_MNIST_data', download=True, train=train, transform=transform)
    return data

def imsave(imgs, titles,label):
    f = plt.figure()
    for img,title in zip(imgs,titles):
        img_t = img.detach().squeeze(0)
        index = [i for i,x in enumerate(imgs) if torch.equal(x, img)]
        f.add_subplot(1,  len(imgs), index[0] + 1)
        plt.imshow(img_t)
        plt.title(title)
    plt.savefig("images/comparisons/comparison_" + str(label) +".png", bbox_inches = "tight")
