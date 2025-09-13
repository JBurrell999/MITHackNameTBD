# server/run_duel.py
import os, io, base64, threading, time, yaml, logging
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

import torch
import gymnasium as gym
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Ensure local imports work ---
from server.models.vae import ConvVAE
from server.models.mdn_rnn import MDNRNN
from server.models.controller import LinearController

# ----------------- Config -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("vmc-duel")

BOUNDS_PATH = os.getenv("BOUNDS_PATH", "configs/bounds.yaml")
with open(BOUNDS_PATH, "r") as f:
    BOUNDS = yaml.safe_load(f)

CHECKPOINT_STEPS = int(os.getenv("CHECKPOINT_STEPS", "50"))  # knob apply cadence
TARGET_HZ = float(os.getenv("TARGET_HZ", "20"))              # sim loop rate
RENDER_JPEG_QUALITY = int(os.getenv("RENDER_JPEG_QUALITY", "70"))
DEVICE = torch.device("cpu")  # keep CPU for hackathon demo
ENV_ID = os.getenv("ENV_ID", "CarRacing-v3")

WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "server/weights")
VAE_PATH = os.path.join(WEIGHTS_DIR, "vae.pth")
MDN_PATH = os.path.join(WEIGHTS_DIR, "mdnrnn.pth")
CTRL_A_PATH = os.path.join(WEIGHTS_DIR, "controllerA.npz")
CTRL_B_PATH = os.path.join(WEIGHTS_DIR, "controllerB.npz")

# ----------------- Utils -----------------
def clamp_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clamp arbitrary param dict against BOUNDS schema.
    Supports simple floats and nested dict for gating_bias.
    """
    out: Dict[str, Any] = {}
    for k, spec in BOUNDS.items():
        if k not in params:
            continue
        v = params[k]
        if isinstance(spec, list) and len(spec) == 2:
            lo, hi = float(spec[0]), float(spec[1])
            try:
                out[k] = float(max(lo, min(hi, float(v))))
            except Exception:
                continue
        elif isinstance(spec, dict):  # e.g., gating_bias
            inner = {}
            for subk, rng in spec.items():
                if subk in v:
                    lo, hi = float(rng[0]), float(rng[1])
                    inner[subk] = float(max(lo, min(hi, float(v[subk]))))
            out[k] = inner
    return out

def npimg_to_b64(img: np.ndarray, quality: int = 70) -> str:
    im = Image.fromarray(img)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

# ----------------- Player & Duel State -----------------
class PlayerState:
    def __init__(self, name: str, controller_weights_path: str, in_dim: int, out_dim: int):
        self.name = name
        self.ctrl = LinearController(in_dim, out_dim)
        self.path = controller_weights_path
        self.knobs = {
            "alpha_action_smoothing": 0.2,
            "risk_aversion": 0.0,
            "emergency_brake_threshold": 0.8,
            "lookahead_horizon": 10,
            "gating_bias": {"rain": 0.0, "drift": 0.0, "straight": 0.0},
        }
        self.pending_knobs: Dict[str, Any] = {}
        self.h = None  # LSTM hidden state tuple (h, c)
        self.total_reward = 0.0
        self.offtrack_time = 0.0
        self.loaded = False

    def load_or_fail(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Controller weights not found: {self.path}")
        w = np.load(self.path)
        self.ctrl.set_params({"W": w["W"], "b": w["b"]})
        self.loaded = True
        LOGGER.info("Loaded controller for %s from %s", self.name, self.path)

class DuelCore:
    def __init__(self, seed: int = 123):
        # Load V and M
        if not os.path.exists(VAE_PATH) or not os.path.exists(MDN_PATH):
            raise RuntimeError("Missing VAE/MDN weights in server/weights/.")
        self.vae = ConvVAE(z_dim=16).to(DEVICE)
        self.vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
        self.vae.eval()

        self.mdn = MDNRNN(z_dim=16, a_dim=3, h_dim=128, K=5).to(DEVICE)
        self.mdn.load_state_dict(torch.load(MDN_PATH, map_location=DEVICE))
        self.mdn.eval()

        in_dim, out_dim = 16 + 128, 3
        self.pA = PlayerState("A", CTRL_A_PATH, in_dim, out_dim)
        self.pB = PlayerState("B", CTRL_B_PATH, in_dim, out_dim)

        # Envs: two single-car envs with the same seed = same track
        self.envA = gym.make(ENV_ID, render_mode="rgb_array")
        self.envB = gym.make(ENV_ID, render_mode="rgb_array")

        self.seed = seed
        self.lock = threading.Lock()
        self.running = False
        self.step_count = 0

        # last frames for telemetry
        self.last_frameA_b64 = ""
        self.last_frameB_b64 = ""

        self.reset(seed=self.seed)

    # ---------- Core helpers ----------
    def _encode_z(self, frame: np.ndarray) -> torch.Tensor:
        im = Image.fromarray(frame).resize((64, 64))
        x = torch.from_numpy(np.array(im).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu, _ = self.vae.encode(x)
        return mu  # 1x16

    def _update_hidden(self, P: PlayerState, z_t: torch.Tensor, a_t: np.ndarray):
        with torch.no_grad():
            z_in = z_t.unsqueeze(1)  # [1,1,16]
            a_in = torch.from_numpy(a_t).float().view(1, 1, -1).to(DEVICE)
            _pi, _mu, _ls, P.h = self.mdn(z_in, a_in, P.h)

    # ---------- Lifecycle ----------
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = int(seed)
        obsA, _ = self.envA.reset(seed=self.seed)
        obsB, _ = self.envB.reset(seed=self.seed)

        # init hidden states
        hdim = self.mdn.rnn.hidden_size
        self.pA.h = (torch.zeros(1, 1, hdim, device=DEVICE), torch.zeros(1, 1, hdim, device=DEVICE))
        self.pB.h = (torch.zeros(1, 1, hdim, device=DEVICE), torch.zeros(1, 1, hdim, device=DEVICE))
        self.pA.total_reward = 0.0
        self.pB.total_reward = 0.0
        self.pA.offtrack_time = 0.0
        self.pB.offtrack_time = 0.0
        self.step_count = 0

        # load controllers (fail loudly if missing; see /bootstrap_controllers to create them)
        try:
            self.pA.load_or_fail()
            self.pB.load_or_fail()
        except FileNotFoundError as e:
            LOGGER.warning("Controller missing: %s", e)

        # prime frames
        fA = self.envA.render()
        fB = self.envB.render()
        self.last_frameA_b64 = npimg_to_b64(fA, quality=RENDER_JPEG_QUALITY)
        self.last_frameB_b64 = npimg_to_b64(fB, quality=RENDER_JPEG_QUALITY)

    def step_once(self):
        if not (self.pA.loaded and self.pB.loaded):
            # Don’t advance a race without controllers
            return False

        frameA = self.envA.render()
        frameB = self.envB.render()

        zA = self._encode_z(frameA).squeeze(0).cpu().numpy()
        zB = self._encode_z(frameB).squeeze(0).cpu().numpy()
        hA = self.pA.h[0].squeeze(0).squeeze(0).detach().cpu().numpy()
        hB = self.pB.h[0].squeeze(0).squeeze(0).detach().cpu().numpy()

        aA = self.pA.ctrl.act(zA, hA, self.pA.knobs)
        aB = self.pB.ctrl.act(zB, hB, self.pB.knobs)

        _obsA, rA, termA, truncA, infoA = self.envA.step(aA)
        _obsB, rB, termB, truncB, infoB = self.envB.step(aB)

        self._update_hidden(self.pA, torch.from_numpy(zA).float().to(DEVICE), aA)
        self._update_hidden(self.pB, torch.from_numpy(zB).float().to(DEVICE), aB)

        self.pA.total_reward += float(rA)
        self.pB.total_reward += float(rB)

        # (CarRacing’s info is sparse; keep placeholders for future features)
        self.pA.offtrack_time += float(infoA.get("off_track", 0.0)) if isinstance(infoA, dict) else 0.0
        self.pB.offtrack_time += float(infoB.get("off_track", 0.0)) if isinstance(infoB, dict) else 0.0

        # update frames for telemetry
        self.last_frameA_b64 = npimg_to_b64(self.envA.render(), quality=RENDER_JPEG_QUALITY)
        self.last_frameB_b64 = npimg_to_b64(self.envB.render(), quality=RENDER_JPEG_QUALITY)

        self.step_count += 1
        if self.step_count % CHECKPOINT_STEPS == 0:
            if self.pA.pending_knobs:
                self.pA.knobs.update(clamp_params(self.pA.pending_knobs))
                self.pA.pending_knobs = {}
            if self.pB.pending_knobs:
                self.pB.knobs.update(clamp_params(self.pB.pending_knobs))
                self.pB.pending_knobs = {}

        done = (termA or truncA or termB or truncB)
        if done:
            LOGGER.info("Episode finished at step=%d (A: %.1f, B: %.1f)",
                        self.step_count, self.pA.total_reward, self.pB.total_reward)
        return done

# ----------------- FastAPI App -----------------
app = FastAPI(title="VMC Duel Server", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

DUEL = DuelCore(seed=int(os.getenv("SEED", "123")))
SIM_THREAD: Optional[threading.Thread] = None

def _sim_loop():
    dt = 1.0 / max(1e-3, TARGET_HZ)
    while DUEL.running:
        t0 = time.time()
        with DUEL.lock:
            done = DUEL.step_once()
            if done:
                DUEL.running = False
        # keep rate modest and CPU stable
        elapsed = time.time() - t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

# ----------------- Schemas & Endpoints -----------------
class HealthOut(BaseModel):
    ok: bool
    env: str
    seed: int
    running: bool
    controllers_loaded: bool

@app.get("/health", response_model=HealthOut)
def health():
    return HealthOut(
        ok=True,
        env=ENV_ID,
        seed=DUEL.seed,
        running=DUEL.running,
        controllers_loaded=(DUEL.pA.loaded and DUEL.pB.loaded),
    )

class ControlReq(BaseModel):
    action: str = Field(..., description="start|pause|reset")
    seed: Optional[int] = None

@app.post("/control")
def control(req: ControlReq):
    global SIM_THREAD
    if req.action == "start":
        if not (DUEL.pA.loaded and DUEL.pB.loaded):
            raise HTTPException(status_code=422, detail="Controllers not loaded. Use /bootstrap_controllers or place controllerA/B.npz.")
        if not DUEL.running:
            DUEL.running = True
            SIM_THREAD = threading.Thread(target=_sim_loop, daemon=True)
            SIM_THREAD.start()
        return {"ok": True, "state": "running"}
    elif req.action == "pause":
        DUEL.running = False
        return {"ok": True, "state": "paused"}
    elif req.action == "reset":
        with DUEL.lock:
            DUEL.running = False
            DUEL.reset(seed=req.seed if req.seed is not None else DUEL.seed)
        return {"ok": True, "state": "reset", "seed": DUEL.seed}
    else:
        raise HTTPException(status_code=400, detail="Unknown action")

class KnobReq(BaseModel):
    player: str  # "A" or "B"
    params: Dict[str, Any]

@app.post("/update_knobs")
def update_knobs(req: KnobReq):
    target = req.player.upper()
    with DUEL.lock:
        if target == "A":
            DUEL.pA.pending_knobs.update(req.params)
        elif target == "B":
            DUEL.pB.pending_knobs.update(req.params)
        else:
            raise HTTPException(status_code=400, detail="player must be 'A' or 'B'")
    return {"ok": True, "applies_in_steps": CHECKPOINT_STEPS}

@app.get("/telemetry")
def telemetry():
    with DUEL.lock:
        return {
            "running": DUEL.running,
            "step": DUEL.step_count,
            "seed": DUEL.seed,
            "playerA": {
                "reward": DUEL.pA.total_reward,
                "offtrack_time": DUEL.pA.offtrack_time,
                "knobs": DUEL.pA.knobs,
                "frame_b64": DUEL.last_frameA_b64
            },
            "playerB": {
                "reward": DUEL.pB.total_reward,
                "offtrack_time": DUEL.pB.offtrack_time,
                "knobs": DUEL.pB.knobs,
                "frame_b64": DUEL.last_frameB_b64
            }
        }

# -------- Optional: controller bootstrap (CMA-ES in dream) --------
class BootstrapReq(BaseModel):
    tauA: float = 1.15
    tauB: float = 1.05
    iters: int = 10
    pop: int = 16
    steps: int = 400

@app.post("/bootstrap_controllers")
def bootstrap_controllers(req: BootstrapReq):
    """
    Trains minimal linear controllers inside a DreamEnv-like loop that uses MDN-RNN state.
    CPU-friendly. Blocks until both A and B are written to server/weights/.
    """
    try:
        import cma  # lazy import
    except Exception:
        raise HTTPException(status_code=500, detail="Missing dependency 'cma'. pip install cma")

    from server.models.dream_env import DreamEnv  # relies on your earlier file

    def train_one(tau: float, save_path: str):
        in_dim, out_dim = 16 + 128, 3
        ctrl = LinearController(in_dim, out_dim)
        z0 = np.zeros(16, np.float32); h0 = np.zeros(128, np.float32)

        def decode(flat):
            W = flat[:out_dim*in_dim].reshape(out_dim, in_dim); b = flat[out_dim*in_dim:]
            return {"W": W, "b": b}

        def fitness(x):
            P = decode(np.array(x))
            ctrl.set_params(P)
            dream = DreamEnv(DUEL.vae, DUEL.mdn, z0, h0, tau=tau, device=DEVICE.type)
            total = 0.0
            for _ in range(req.steps):
                z, h = dream.get_latent()
                a = ctrl.act(z, h, {"alpha_action_smoothing": 0.2})
                _, r, _ = dream.step(a)
                total += r
            return -total  # CMA-ES minimizes

        x0 = np.concatenate([ctrl.W.flatten(), ctrl.b])
        es = cma.CMAEvolutionStrategy(x0, 0.2, {'popsize': req.pop})
        for _ in range(req.iters):
            xs = es.ask()
            es.tell(xs, [fitness(s) for s in xs])
        best = np.array(es.best.x)
        P = decode(best)
        np.savez(save_path, **P)

    # pause sim during bootstrap
    with DUEL.lock:
        DUEL.running = False

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    train_one(req.tauA, CTRL_A_PATH)
    train_one(req.tauB, CTRL_B_PATH)

    # reload on reset
    with DUEL.lock:
        DUEL.reset(seed=DUEL.seed)
    return {"ok": True, "msg": "Controllers trained and loaded."}

# -------- Dev runner --------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.run_duel:app", host="0.0.0.0", port=8000, reload=False)
