import os, glob, argparse, numpy as np, torch
from tqdm import tqdm
from server.models.vae import ConvVAE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/buffer")
    ap.add_argument("--vae", default="server/weights/vae.pth")
    ap.add_argument("--z_out", default="data/buffer/latents")
    args = ap.parse_args()

    os.makedirs(args.z_out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = ConvVAE(z_dim=16).to(device)
    vae.load_state_dict(torch.load(args.vae, map_location=device))
    vae.eval()

    eps = sorted(glob.glob(os.path.join(args.data, "ep_*")))
    for ep_dir in eps:
        out_ep = os.path.join(args.z_out, os.path.basename(ep_dir))
        os.makedirs(out_ep, exist_ok=True)
        frames = sorted(glob.glob(os.path.join(ep_dir, "frame_*.npy")))
        for fpath in tqdm(frames, desc=os.path.basename(ep_dir)):
            x = np.load(fpath).astype(np.float32) / 255.0
            x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).to(device)
            with torch.no_grad():
                mu, logvar = vae.encode(x)
                z = mu  # use mean as latent
            np.save(os.path.join(out_ep, os.path.basename(fpath).replace("frame", "z")), z.squeeze(0).cpu().numpy())

if __name__ == "__main__":
    main()
