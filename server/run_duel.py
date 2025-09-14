import traceback
import os, io, time, base64, threading, logging
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image
import yaml

import torch
import gymnasium as gym
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---- Local model imports ----
from server.models.vae import ConvVAE
from server.models.mdn_rnn import MDNRNN
from server.models.controller import LinearController

# ----------------- Config -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("vmc-duel")

BOUNDS_PATH = os.getenv("BOUNDS_PATH", "configs/bounds.yaml")
with open(BOUNDS_PATH, "r") as f:
    BOUNDS = yaml.safe_load(f)

CHECKPOINT_STEPS = int(os.getenv("CHECKPOINT_STEPS", "50"))
TARGET_HZ = float(os.getenv("TARGET_HZ", "20"))
RENDER_JPEG_QUALITY = int(os.getenv("RENDER_JPEG_QUALITY", "70"))
DEVICE = torch.device("cpu")
ENV_ID = os.getenv("ENV_ID", "CarRacing-v3")

WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "server/weights")
VAE_PATH = os.path.join(WEIGHTS_DIR, "vae.pth")
MDN_PATH = os.path.join(WEIGHTS_DIR, "mdnrnn.pth")
CTRL_A_PATH = os.path.join(WEIGHTS_DIR, "controllerA.npz")
CTRL_B_PATH = os.path.join(WEIGHTS_DIR, "controllerB.npz")

# ----------------- Utils -----------------
def clamp_params(params: Dict[str, Any]) -> Dict[str, Any]:
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
        elif isinstance(spec, dict):  # nested e.g. gating_bias
            inner = {}
            for subk, rng in spec.items():
                if isinstance(v, dict) and subk in v:
                    lo, hi = float(rng[0]), float(rng[1])
                    inner[subk] = float(max(lo, min(hi, float(v[subk]))))
            out[k] = inner
    return out

def npimg_to_b64(img: np.ndarray, quality: int = 70) -> str:
    im = Image.fromarray(img)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

# ---- Action helpers ----
def _ema(prev: np.ndarray, cur: np.ndarray, alpha: float) -> np.ndarray:
    prev = np.asarray(prev, dtype=np.float32).reshape(3,)
    cur  = np.asarray(cur,  dtype=np.float32).reshape(3,)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * cur + (1.0 - alpha) * prev

def _squash_action(a, min_gas: float = 0.05, brake_deadzone: float = 0.02) -> np.ndarray:
    """
    Map raw controller outputs to valid CarRacing action space:
      steer in [-1,1] via tanh
      gas   in [0,1] via sigmoid (then floor at min_gas)
      brake in [0,1] via sigmoid (then apply deadzone and no gas+brake simultaneously)
    """
    a = np.asarray(a, dtype=np.float32).reshape(3,)
    steer = np.tanh(a[0])                 # [-1, 1]
    gas   = 1.0 / (1.0 + np.exp(-a[1]))   # (0, 1)
    brake = 1.0 / (1.0 + np.exp(-a[2]))   # (0, 1)

    # keep it rolling
    if gas < min_gas:
        gas = min_gas

    # kill tiny brake that cancels gas
    if brake < brake_deadzone:
        brake = 0.0

    # never gas & brake at once (prefer gas)
    if gas > 0.05 and brake > 0.0:
        brake = 0.0

    return np.array([steer, gas, brake], dtype=np.float32)

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
            # expose safe defaults so UI or API can tune if needed
            "min_gas": 0.05,          # baseline throttle to avoid stalls
            "brake_deadzone": 0.02,   # ignore tiny brake that cancels gas
        }
        self.pending_knobs: Dict[str, Any] = {}
        self.h = None  # (h, c) for MDN-RNN
        self.total_reward = 0.0
        self.offtrack_time = 0.0
        self.loaded = False
        self.last_a = np.zeros(3, dtype=np.float32)  # for EMA smoothing

    def load_or_fail(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Controller weights not found: {self.path}")
        w = np.load(self.path)
        self.ctrl.set_params({"W": w["W"], "b": w["b"]})
        self.loaded = True
        LOGGER.info("Loaded controller for %s from %s", self.name, self.path)

class DuelCore:
    def __init__(self, seed: int = 123):
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

        # Use observations; CarRacing obs ~96x96x3
        self.envA = gym.make(ENV_ID, render_mode=None)
        self.envB = gym.make(ENV_ID, render_mode=None)

        self.seed = seed
        self.lock = threading.Lock()
        self.running = False
        self.step_count = 0

        self.last_obsA: Optional[np.ndarray] = None
        self.last_obsB: Optional[np.ndarray] = None
        self.last_frameA_b64 = ""
        self.last_frameB_b64 = ""

        self.reset(seed=self.seed)
        self.last_error = None

    # ---------- Core helpers ----------
    def _encode_z(self, frame: np.ndarray) -> torch.Tensor:
        # frame: HxWxC uint8 -> 64x64 for VAE
        im = Image.fromarray(frame).resize((64, 64))
        x = torch.from_numpy(np.array(im).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mu, _ = self.vae.encode(x)  # [1,16]
        return mu

    def _update_hidden(self, P: PlayerState, z_t: torch.Tensor, a_t: np.ndarray):
        with torch.no_grad():
            # z -> torch [1, 16]
            if not isinstance(z_t, torch.Tensor):
                z_in = torch.as_tensor(z_t, dtype=torch.float32, device=DEVICE)
            else:
                z_in = z_t
            if z_in.dim() == 3:          # [S,B,F] -> squeeze seq
                z_in = z_in.squeeze(0)
            z_in = z_in.view(1, -1)      # [1, 16]

            # a -> torch [1, 3]
            a_in = torch.as_tensor(a_t, dtype=torch.float32, device=DEVICE).view(1, -1)  # [1, 3]

            # call MDN-RNN (expects z: [1,16], a: [1,3])
            _pi, _mu, _ls, P.h = self.mdn(z_in, a_in, P.h)

    # ---------- Lifecycle ----------
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = int(seed)
        obsA, _ = self.envA.reset(seed=self.seed)
        obsB, _ = self.envB.reset(seed=self.seed)

        hdim = self.mdn.rnn.hidden_size
        self.pA.h = (torch.zeros(1, 1, hdim, device=DEVICE), torch.zeros(1, 1, hdim, device=DEVICE))
        self.pB.h = (torch.zeros(1, 1, hdim, device=DEVICE), torch.zeros(1, 1, hdim, device=DEVICE))
        self.pA.total_reward = 0.0
        self.pB.total_reward = 0.0
        self.pA.offtrack_time = 0.0
        self.pB.offtrack_time = 0.0
        self.step_count = 0

        self.pA.last_a = np.zeros(3, dtype=np.float32)
        self.pB.last_a = np.zeros(3, dtype=np.float32)

        try:
            self.pA.load_or_fail()
            self.pB.load_or_fail()
        except FileNotFoundError as e:
            LOGGER.warning("Controller missing: %s", e)

        self.last_obsA = obsA
        self.last_obsB = obsB
        self.last_frameA_b64 = npimg_to_b64(self.last_obsA, quality=RENDER_JPEG_QUALITY)
        self.last_frameB_b64 = npimg_to_b64(self.last_obsB, quality=RENDER_JPEG_QUALITY)

    def step_once(self):
        if not (self.pA.loaded and self.pB.loaded):
            return False

        # Encode from last observations
        zA_t = self._encode_z(self.last_obsA)   # torch [1,16]
        zB_t = self._encode_z(self.last_obsB)   # torch [1,16]
        zA_np = zA_t.squeeze(0).cpu().numpy()   # [16] for controller
        zB_np = zB_t.squeeze(0).cpu().numpy()   # [16] for controller

        hA_np = self.pA.h[0].squeeze(0).squeeze(0).detach().cpu().numpy()  # [128]
        hB_np = self.pB.h[0].squeeze(0).squeeze(0).detach().cpu().numpy()  # [128]

        # --- Act (raw -> EMA smoothing -> squash to valid range) ---
        aA_raw = self.pA.ctrl.act(zA_np, hA_np, self.pA.knobs)
        aB_raw = self.pB.ctrl.act(zB_np, hB_np, self.pB.knobs)

        alphaA = float(self.pA.knobs.get("alpha_action_smoothing", 0.2))
        alphaB = float(self.pB.knobs.get("alpha_action_smoothing", 0.2))

        # EMA smoothing in raw space (reduces jitter)
        aA_smooth = _ema(self.pA.last_a, aA_raw, alphaA)
        aB_smooth = _ema(self.pB.last_a, aB_raw, alphaB)
        self.pA.last_a = aA_smooth
        self.pB.last_a = aB_smooth

        # squash to env-legal space with safety nets
        aA = _squash_action(
            aA_smooth,
            min_gas=float(self.pA.knobs.get("min_gas", 0.05)),
            brake_deadzone=float(self.pA.knobs.get("brake_deadzone", 0.02)),
        )
        aB = _squash_action(
            aB_smooth,
            min_gas=float(self.pB.knobs.get("min_gas", 0.05)),
            brake_deadzone=float(self.pB.knobs.get("brake_deadzone", 0.02)),
        )

        # Step envs (obs, reward, done, trunc, info)
        obsA, rA, termA, truncA, infoA = self.envA.step(aA.astype(np.float32))
        obsB, rB, termB, truncB, infoB = self.envB.step(aB.astype(np.float32))

        # Update MDN hidden — pass z WITHOUT squeeze; normalizer handles shapes
        self._update_hidden(self.pA, zA_t, aA)
        self._update_hidden(self.pB, zB_t, aB)

        self.pA.total_reward += float(rA)
        self.pB.total_reward += float(rB)
        self.pA.offtrack_time += float(infoA.get("off_track", 0.0)) if isinstance(infoA, dict) else 0.0
        self.pB.offtrack_time += float(infoB.get("off_track", 0.0)) if isinstance(infoB, dict) else 0.0

        # Update last obs & thumbnails for UI
        self.last_obsA = obsA
        self.last_obsB = obsB
        self.last_frameA_b64 = npimg_to_b64(obsA, quality=RENDER_JPEG_QUALITY)
        self.last_frameB_b64 = npimg_to_b64(obsB, quality=RENDER_JPEG_QUALITY)

        # Knob checkpoints
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
app = FastAPI(title="VMC Duel Server", version="1.2")
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
        try:
            with DUEL.lock:
                done = DUEL.step_once()
                if done:
                    DUEL.running = False
        except Exception:
            DUEL.last_error = traceback.format_exc()
            DUEL.running = False
            LOGGER.error("Sim crashed:\n%s", DUEL.last_error)
        elapsed = time.time() - t0
        if elapsed < dt and DUEL.running:
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
            "error": DUEL.last_error,
            "playerA": {
                "reward": DUEL.pA.total_reward,
                "offtrack_time": DUEL.pA.offtrack_time,
                "knobs": DUEL.pA.knobs,
                "frame_b64": DUEL.last_frameA_b64,
            },
            "playerB": {
                "reward": DUEL.pB.total_reward,
                "offtrack_time": DUEL.pB.offtrack_time,
                "knobs": DUEL.pB.knobs,
                "frame_b64": DUEL.last_frameB_b64,
            },
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
    try:
        import cma
    except Exception:
        raise HTTPException(status_code=500, detail="Missing dependency 'cma'. pip install cma")

    try:
        from server.models.dream_env import DreamEnv
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Missing DreamEnv (server/models/dream_env.py). {e}")

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
            return -total

        x0 = np.concatenate([ctrl.W.flatten(), ctrl.b])
        es = cma.CMAEvolutionStrategy(x0, 0.2, {'popsize': req.pop})
        for _ in range(req.iters):
            xs = es.ask()
            es.tell(xs, [fitness(s) for s in xs])
        best = np.array(es.best.x)
        P = decode(best)
        np.savez(save_path, **P)

    with DUEL.lock:
        DUEL.running = False

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    train_one(req.tauA, CTRL_A_PATH)
    train_one(req.tauB, CTRL_B_PATH)

    with DUEL.lock:
        DUEL.reset(seed=DUEL.seed)
    return {"ok": True, "msg": "Controllers trained and loaded."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.run_duel:app", host="0.0.0.0", port=8000, reload=False)
