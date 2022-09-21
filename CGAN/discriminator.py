
import torch.nn as nn
import torch 
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, n_classes, img_shape):
        super().__init__()
        
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, n_classes)
        

        def d_block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(0.3))
            return layers

        self.model = nn.Sequential(
            *d_block(n_classes + int(np.prod(img_shape)), 1024),
            *d_block(1024, 512),
            *d_block(512, 256),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        x = torch.cat([x, self.label_emb(labels)], 1)
        out = self.model(x)
        return out.squeeze()