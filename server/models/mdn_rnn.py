# server/models/mdn_rnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MDNRNN(nn.Module):
    """
    Minimal MDN-RNN used in world models:
      - RNN: single-layer LSTM with hidden size h_dim
      - Heads over last hidden state: pi, mu, log_sigma
    Notes:
      - We only need consistent module names so your .pth loads:
        rnn.*, pi.*, mu.*, log_sigma.*
      - Forward accepts z and a in 1D/2D/3D and adapts shapes.
    """
    def __init__(self, z_dim: int = 16, a_dim: int = 3, h_dim: int = 128, K: int = 5, batch_first: bool = True):
        super().__init__()
        self.z_dim = z_dim
        self.a_dim = a_dim
        self.h_dim = h_dim
        self.K = K
        self.batch_first = batch_first

        self.rnn = nn.LSTM(input_size=z_dim + a_dim, hidden_size=h_dim, num_layers=1, batch_first=batch_first)

        # MDN heads from hidden -> mixture over next latent z (K mixtures)
        # (Exact usage isn’t critical for the duel loop, but names/shapes must match state_dict.)
        self.pi = nn.Linear(h_dim, K)                     # [*, K]
        self.mu = nn.Linear(h_dim, K * z_dim)            # [*, K*z_dim]
        self.log_sigma = nn.Linear(h_dim, K * z_dim)     # [*, K*z_dim]

    def _to_2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize arbitrary rank input to [B=1, F]
        Accepts: [F], [B,F], [S,B,F] -> returns [1,F] (takes the first slice for higher ranks).
        """
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
        else:
            x = x.to(next(self.parameters()).device, dtype=torch.float32)

        if x.dim() == 3:   # [S,B,F] or [B,T,F] — grab the first step
            x = x[0]  # -> [B,F]
        if x.dim() == 1:   # [F]
            x = x.unsqueeze(0)  # [1,F]
        elif x.dim() != 2:
            x = x.view(1, -1)   # fallback
        return x  # [1,F] if B==1, otherwise [B,F] (we keep B if not 1)

    def forward(self, z, a, h=None):
        """
        z: [F] or [B,F] or [S,B,F]
        a: [F] or [B,F] or [S,B,F]
        h: (h0, c0) with shape [num_layers=1, batch, h_dim]
        Returns: (pi, mu, log_sigma, h)
          - pi:         [B, T=1, K]        if batch_first else [T=1, B, K]
          - mu:         [B, T=1, K*z_dim]  if batch_first else [T=1, B, K*z_dim]
          - log_sigma:  [B, T=1, K*z_dim]  if batch_first else [T=1, B, K*z_dim]
        """
        z2 = self._to_2d(z)  # [B,Fz] (B defaults to 1)
        a2 = self._to_2d(a)  # [B,Fa]

        # If batch sizes disagree, broadcast the singleton one
        B = max(z2.size(0), a2.size(0))
        if z2.size(0) != B:
            if z2.size(0) == 1:
                z2 = z2.expand(B, z2.size(1))
            else:
                raise ValueError(f"Batch mismatch for z: {z2.shape} vs a: {a2.shape}")
        if a2.size(0) != B:
            if a2.size(0) == 1:
                a2 = a2.expand(B, a2.size(1))
            else:
                raise ValueError(f"Batch mismatch for z: {z2.shape} vs a: {a2.shape}")

        x2 = torch.cat([z2, a2], dim=-1)  # [B, Fz+Fa]

        # Make one-step 3D sequence for LSTM
        if self.batch_first:
            x3 = x2.unsqueeze(1)          # [B, T=1, Fin]
        else:
            x3 = x2.unsqueeze(0)          # [T=1, B, Fin]

        y, h = self.rnn(x3, h)            # y: [B,1,H] or [1,B,H] depending on batch_first

        # Heads applied on time dimension (same shape as y)
        pi = self.pi(y)
        mu = self.mu(y)
        log_sigma = self.log_sigma(y)
        return pi, mu, log_sigma, h

# Optional MDN loss alias, if other code expects it
def mdn_loss(*args, **kwargs):
    # Placeholder to keep imports happy; your training code may not call this in the duel.
    raise NotImplementedError("mdn_loss is not used by run_duel; provide your training loss if needed.")

__all__ = ["MDNRNN", "mdn_loss"]
