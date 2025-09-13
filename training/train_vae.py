import os, glob, argparse, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from server.models.vae import ConvVAE, vae_loss

class FrameDataset(Dataset):
    def __init__(self, root):
        self.frames = sorted(glob.glob(os.path.join(root, "ep_*", "frame_*.npy")))
    def __len__(self): return len(self.frames)
    def __getitem__(self, i):
        x = np.load(self.frames[i]).astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2,0,1)  # C,H,W
        return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/buffer")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--z_dim", type=int, default=16)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--save", default="server/weights/vae.pth")
    args = ap.parse_args()

    ds = FrameDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = ConvVAE(z_dim=args.z_dim).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)

    vae.train()
    for ep in range(args.epochs):
        for i, x in enumerate(dl):
            x = x.to(device)
            xhat, mu, logvar = vae(x)
            loss, logs = vae_loss(x, xhat, mu, logvar, beta=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch {ep+1} | loss={loss.item():.4f} recon={logs['recon']:.4f} kld={logs['kld']:.4f}")
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save(vae.state_dict(), args.save)

if __name__ == "__main__":
    main()
