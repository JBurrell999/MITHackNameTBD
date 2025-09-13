import os, glob, argparse, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from server.models.mdn_rnn import MDNRNN, mdn_loss

class ZActDataset(Dataset):
    def __init__(self, z_root, act_root):
        self.ep_dirs = sorted(glob.glob(os.path.join(z_root, "ep_*")))
        self.act_root = act_root
    def __len__(self): return len(self.ep_dirs)
    def __getitem__(self, i):
        ep_z = sorted(glob.glob(os.path.join(self.ep_dirs[i], "z_*.npy")))
        T = len(ep_z) - 1
        z = np.stack([np.load(p) for p in ep_z], axis=0).astype(np.float32)
        # actions aligned: action_t produces next_z at t+1
        ep_id = os.path.basename(self.ep_dirs[i])
        acts = sorted(glob.glob(os.path.join(self.act_root, ep_id, "action_*.npy")))
        a = np.stack([np.load(p) for p in acts], axis=0).astype(np.float32)
        # trim to min length
        T = min(T, a.shape[0])
        z_in, a_in, z_next = z[:T], a[:T], z[1:T+1]
        return z_in, a_in, z_next

def collate(batch):
    # variable lengths are rare here; pad if needed
    zs, as_, znexts = zip(*batch)
    maxT = min(min(x.shape[0] for x in zs), 256)
    Z = torch.tensor(np.stack([z[:maxT] for z in zs]))        # [B,T,D]
    A = torch.tensor(np.stack([a[:maxT] for a in as_]))       # [B,T,Ad]
    ZN= torch.tensor(np.stack([zn[:maxT] for zn in znexts]))  # [B,T,D]
    return Z, A, ZN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z_dir", default="data/buffer/latents")
    ap.add_argument("--act_dir", default="data/buffer")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--mixtures", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--save", default="server/weights/mdnrnn.pth")
    args = ap.parse_args()

    ds = ZActDataset(args.z_dir, args.act_dir)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MDNRNN(z_dim=16, a_dim=3, h_dim=args.hidden, K=args.mixtures).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for ep in range(args.epochs):
        for Z, A, ZN in dl:
            Z,A,ZN = Z.to(device), A.to(device), ZN.to(device)
            pi, mu, log_sigma, _ = model(Z, A, None)
            loss = mdn_loss(ZN, pi, mu, log_sigma)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch {ep+1} | mdn_loss={loss.item():.4f}")
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save(model.state_dict(), args.save)

if __name__ == "__main__":
    main()
