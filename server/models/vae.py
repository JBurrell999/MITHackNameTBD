import torch, torch.nn as nn, torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, z_dim=16):
        super().__init__()
        # Encoder: 64x64x3 -> z
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),   # 32x32
            nn.Conv2d(32,64, 4, 2, 1), nn.ReLU(),   # 16x16
            nn.Conv2d(64,128,4, 2, 1), nn.ReLU(),   # 8x8
            nn.Conv2d(128,256,4,2,1), nn.ReLU(),    # 4x4
        )
        self.fc_mu = nn.Linear(256*4*4, z_dim)
        self.fc_logvar = nn.Linear(256*4*4, z_dim)

        # Decoder: z -> 64x64x3
        self.fc = nn.Linear(z_dim, 256*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(), # 8x8
            nn.ConvTranspose2d(128,64, 4,2,1), nn.ReLU(), # 16x16
            nn.ConvTranspose2d(64, 32, 4,2,1), nn.ReLU(), # 32x32
            nn.ConvTranspose2d(32, 3,  4,2,1), nn.Sigmoid() # 64x64
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu, logvar):
        std = (0.5*logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.fc(z).view(-1,256,4,4)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar

def vae_loss(x, xhat, mu, logvar, beta=1.0):
    recon = F.mse_loss(xhat, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta*kld, {'recon': recon.item(), 'kld': kld.item()}
