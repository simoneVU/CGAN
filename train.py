from dataset.fashionMNIST import load_F_MNIST
from CGAN.discriminator import Discriminator
from CGAN.generator import Generator

import torch
import torch.nn as nn
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt

import argparse
from tqdm import tqdm
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=250, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--latent_dim", type=int, default=100, help="dim. of latent space")
parser.add_argument("--img_dim", type=int, default=28, help="size of both image dimensions")
parser.add_argument("--num_classes", type=int, default=10, help="nr. of classes")

args = parser.parse_args()

def g_step(optimizer, batch_size, latent_dim, n_classes, generator,loss):
    optimizer.zero_grad()

    #Generate noise samples and labels for the generator
    noise = torch.randn((batch_size, latent_dim))
    fake_labels = torch.LongTensor(np.random.randint(0, n_classes, batch_size))

    # Generate a batch of fake images
    fake_images = generator(noise, fake_labels)

    # g_loss measures the generator's ability to fool the discriminator
    validity = discriminator(fake_images, fake_labels)
    g_loss = loss(validity, torch.ones(batch_size, requires_grad=False))

    g_loss.backward()
    optimizer.step()
    return g_loss.item()


def d_step(optimizer, batch_size, real_images, labels, n_classes, discriminator, loss):    
    optimizer.zero_grad()
    
    #Calculate loss for real images
    real_loss = loss(discriminator(real_images, labels),torch.ones(batch_size, requires_grad=False))

    #Calculate loss for fake images

    #Generate noise samples and labels for the generator
    noise = torch.randn((batch_size, latent_dim))
    fake_labels = torch.LongTensor(np.random.randint(0, n_classes, batch_size))

    # Generate a batch of fake images
    fake_images = generator(noise, fake_labels)
    fake_loss = loss(discriminator(fake_images.detach(), fake_labels), torch.zeros(batch_size))

    # Total discriminator loss
    total_loss = (real_loss + fake_loss) / 2

    total_loss.backward()
    optimizer.step()

    return total_loss.item()

cuda = True if torch.cuda.is_available() else False
trainset = torch.utils.data.DataLoader(load_F_MNIST(), batch_size=args.batch_size, shuffle=True)

# Loss functions
adversarial_loss = nn.BCELoss()

# Initialize generator and discriminator
latent_dim = args.latent_dim
n_classes = args.num_classes
image_shape = [args.img_dim, args.img_dim]

generator = Generator(latent_dim, n_classes, image_shape)
discriminator = Discriminator(n_classes, image_shape)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)

#Training Loop
n_epochs = args.n_epochs
d_losses = []
g_losses = []

for epoch in tqdm(range(n_epochs)):
    print('Starting epoch {}...'.format(epoch))
    for i, (images, labels) in tqdm(enumerate(trainset), total=len(trainset)):
        real_img = images
        labels = labels
        generator.train()
        batch_size = real_img.size(0)

        d_loss = d_step(d_optimizer, batch_size, real_img, labels, n_classes, discriminator, adversarial_loss)

        g_loss = g_step(g_optimizer, batch_size, latent_dim, n_classes, generator, adversarial_loss)

    generator.eval()
    d_losses.append(d_loss)
    g_losses.append(g_loss)

    print(
        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, n_epochs, i, len(trainset), d_loss, g_loss)
        )

    # Validate every 10 epochs
    if epoch % 10 == 0:
        torch.save({
                'epoch': epoch,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': g_optimizer.state_dict(),
                'loss': g_loss,
            }, "saved_models/checkpoints/generator" + str(epoch) + '.pth')

        z = torch.randn(10, 100)
        labels = torch.LongTensor(np.arange(10))
        sample_images = generator(z, labels).unsqueeze(1).data.cpu()
        save_image(sample_images, "images/generated_steps/%d.png" % epoch , nrow=10, normalize=True) 


# Saving the models
torch.save(generator.state_dict(), "saved_models/generator_weights.pth")
torch.save(discriminator.state_dict(), "saved_models/discriminator_weights.pth")

#Saving loss arrays
with open('losses/d_losses.pkl', 'wb') as f:
   pickle.dump(d_losses, f)

with open('losses/g_losses.pkl', 'wb') as f:
   pickle.dump(g_losses, f)

#Saving loss arrays plot
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.title('Loss of Descriminator and Generator')


plt.plot(range(n_epochs), d_losses, color = "orange",  label = "Discriminator Loss")
plt.plot(range(n_epochs), g_losses, color = "blue",  label = "Generator Loss")
plt.legend()
plt.savefig("images/loss_curve.png", bbox_inches = "tight")
