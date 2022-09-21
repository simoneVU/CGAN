import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, lat_dim, n_classes, img_shape):
        super(Generator, self).__init__()

        self.img_shape = img_shape
        self.latent_dim = lat_dim
        self.label_emb = nn.Embedding(n_classes, n_classes)
        
        def g_block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(0.3))
            return layers

        self.model = nn.Sequential(
            *g_block(lat_dim + n_classes, 256),
            *g_block(256, 512),
            *g_block(512, 1024),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        noise = noise.view(noise.size(0), self.latent_dim)
        x = torch.cat([noise, self.label_emb(labels)], 1)
        out = self.model(x)
        return out.view(x.size(0), *self.img_shape)