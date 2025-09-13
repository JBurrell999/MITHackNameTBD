import numpy as np, torch

class DreamEnv:
    def __init__(self, vae, mdnrnn, z0, h0, tau=1.15, device="cpu"):
        self.vae, self.mdn, self.tau = vae, mdnrnn, tau
        self.device = device
        self.z = torch.tensor(z0, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # [1,1,D]
        self.h = (torch.zeros(1,1,mdnrnn.rnn.hidden_size, device=device),
                  torch.zeros(1,1,mdnrnn.rnn.hidden_size, device=device))

    def step(self, a_np):
        a = torch.tensor(a_np, dtype=torch.float32, device=self.device).view(1,1,-1)  # [1,1,A]
        with torch.no_grad():
            pi, mu, log_sigma, self.h = self.mdn(self.z, a, self.h)  # one-step
            # Gumbel trick on pi, scale log_sigma by tau
            pi = torch.softmax(pi.squeeze(0).squeeze(0), dim=-1)     # [K]
            k = torch.distributions.Categorical(pi).sample()
            mu_k = mu.squeeze(0).squeeze(0)[k]                       # [D]
            log_sigma_k = log_sigma.squeeze(0).squeeze(0)[k] * self.tau
            eps = torch.randn_like(mu_k)
            z_next = mu_k + eps * torch.exp(log_sigma_k)
        self.z = z_next.view(1,1,-1)
        # simple surrogate reward: forward progress proxy = -|z|_perp, penalty for big h-norm
        reward = float(-0.01 * torch.norm(self.h[0]).item())
        done = False
        return z_next.cpu().numpy(), reward, done

    def get_latent(self):
        return self.z.squeeze(0).squeeze(0).detach().cpu().numpy(), \
               self.h[0].squeeze(0).detach().cpu().numpy()
