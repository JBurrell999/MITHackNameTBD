# Pull request here.
import torch, torch.nn as nn, torch.nn.functional as F

def mdn_loss(y, pi, mu, log_sigma):
    # y: [B,T,D]; pi: [B,T,K]; mu/log_sigma: [B,T,K,D]
    # mixture of diagonal Gaussians
    B,T,D = y.size()
    K = pi.size(-1)
    y = y.unsqueeze(2).expand(B,T,K,D)
    log_prob = -0.5 * (((y - mu) / torch.exp(log_sigma))**2).sum(dim=-1) \
               - log_sigma.sum(dim=-1) - 0.5*D*torch.log(torch.tensor(2*3.14159265, device=y.device))
    log_mix = torch.log_softmax(pi, dim=-1)
    log_sum = torch.logsumexp(log_mix + log_prob, dim=-1)  # [B,T]
    return -log_sum.mean()

class MDNRNN(nn.Module):
    def __init__(self, z_dim=16, a_dim=3, h_dim=128, K=5):
        super().__init__()
        self.rnn = nn.LSTM(z_dim + a_dim, h_dim, batch_first=True)
        self.pi = nn.Linear(h_dim, K)
        self.mu = nn.Linear(h_dim, K*z_dim)
        self.log_sigma = nn.Linear(h_dim, K*z_dim)
        self.z_dim, self.K = z_dim, K

    def forward(self, z, a, h=None):
        # z: [B,T,z_dim], a: [B,T,a_dim]
        x = torch.cat([z, a], dim=-1)
        y, h = self.rnn(x, h)     # y: [B,T,h_dim]
        pi = self.pi(y)           # [B,T,K]
        mu = self.mu(y).view(y.size(0), y.size(1), self.K, self.z_dim)
        log_sigma = self.log_sigma(y).view(y.size(0), y.size(1), self.K, self.z_dim)
        return pi, mu, log_sigma, h
