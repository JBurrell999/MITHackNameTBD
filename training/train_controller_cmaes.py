import os, glob, argparse, numpy as np, torch, cma, sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import ConvVAE
from server import MDNRNN
from server import LinearController
from server import DreamEnv


def rollout_return(ctrl, dream_env, steps=600, knobs=None):
    knobs = knobs or {}
    total = 0.0
    for t in range(steps):
        z, h = dream_env.get_latent()
        z = np.squeeze(z)
        h = np.squeeze(h)
        a = ctrl.act(z, h, knobs)
        _, r, done = dream_env.step(a)
        total += r
        if done:
            break
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="server/weights/vae.pth")
    ap.add_argument("--mdn", default="server/weights/mdnrnn.pth")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--tau", type=float, default=1.15)
    ap.add_argument("--save", default="server/weights/controllerA.npz")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = ConvVAE(z_dim=16).to(device)
    vae.load_state_dict(torch.load(args.vae, map_location=device))
    vae.eval()
    mdn = MDNRNN(z_dim=16, a_dim=3, h_dim=128, K=5).to(device)
    mdn.load_state_dict(torch.load(args.mdn, map_location=device))
    mdn.eval()

    in_dim, out_dim = 16 + 128, 3
    ctrl = LinearController(in_dim, out_dim)

    # Random z0 to start dream (or encode a real frame)
    z0 = np.zeros(16, dtype=np.float32)
    h0 = np.zeros(128, dtype=np.float32)
    dream = DreamEnv(vae, mdn, z0, h0, tau=args.tau, device=device.type)
    knobs = {"alpha_action_smoothing": 0.2}

    # CMA-ES over flattened params
    x0 = np.concatenate([ctrl.W.flatten(), ctrl.b])
    es = cma.CMAEvolutionStrategy(x0, 0.2, {"popsize": args.pop})

    def decode(x):
        W = x[: out_dim * in_dim].reshape(out_dim, in_dim)
        b = x[out_dim * in_dim :]
        return {"W": W, "b": b}

    for it in tqdm(range(args.iters)):
        solutions = es.ask()
        scores = []
        for s in solutions:
            ctrl.set_params(decode(np.array(s)))
            # fresh dream each eval
            dream = DreamEnv(vae, mdn, z0, h0, tau=args.tau, device=device.type)
            R = rollout_return(ctrl, dream, steps=600, knobs=knobs)
            scores.append(-R)  # CMA-ES minimizes
        es.tell(solutions, scores)
        es.disp()

    best = decode(es.best.x)
    np.savez(args.save, **best)
    print("Saved controller to", args.save)


if __name__ == "__main__":
    main()
