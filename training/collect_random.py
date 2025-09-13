import os, argparse, numpy as np, gymnasium as gym
from PIL import Image

def downsample(frame, size=64):
    # frame HxWxC uint8 -> size x size RGB
    return np.array(Image.fromarray(frame).resize((size, size)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="CarRacing-v2")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--frames", type=int, default=800)     # per episode cap
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--out", default="data/buffer")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    env = gym.make(args.env, render_mode="rgb_array")
    idx = 0

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        ep_dir = os.path.join(args.out, f"ep_{ep:04d}")
        os.makedirs(ep_dir, exist_ok=True)
        for t in range(args.frames):
            # random or scripted actions; CarRacing actions: [steer, gas, brake] in [-1..1, 0..1, 0..1]
            a = env.action_space.sample()
            frame = env.render()
            f64 = downsample(frame, args.size).astype(np.uint8)
            np.save(os.path.join(ep_dir, f"frame_{t:05d}.npy"), f64)
            np.save(os.path.join(ep_dir, f"action_{t:05d}.npy"), a.astype(np.float32))

            obs, rew, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            if done: break
            idx += 1
    env.close()

if __name__ == "__main__":
    main()
