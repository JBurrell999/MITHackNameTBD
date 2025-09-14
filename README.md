# VMC Duel

Interactive World-Model Car Racing Duel

This project showcases an **end-to-end World Model system** where two AI controllers compete head-to-head in the CarRacing-v3 environment. You can **watch races live, adjust high-level strategy knobs in real time, and explore how learned policies adapt.**

Inspired by the *World Models* framework by David Ha and Jürgen Schmidhuber

---

## Why This Matters

- **Head-to-Head Evaluation**  
  Instead of evaluating agents in isolation, this duel system compares **two learned controllers side by side** under identical conditions.  

- **End-to-End World Models**  
  - **VAE** — compresses raw 96×96 RGB frames into latent vectors  
  - **MDN-RNN** — models temporal dynamics in latent space  
  - **Controller** — linear policy over latent + hidden states  

- **Interactive**  
  A Streamlit UI allows users and judges to adjust knobs like risk aversion or action smoothing and immediately see the difference.  

- **Relevant**  
  Competition between models is a proxy for real-world AI-vs-AI scenarios (autonomous driving, markets, security).  

---

## System Architecture

CarRacing-v3 (x2) --> VAE (encode frames) --> MDN-RNN (temporal model) --> Controller (policy) --> Actions

Two independent pipelines (Player A and Player B) run in parallel.  
A FastAPI backend manages simulation, telemetry, and frames.  
A Streamlit frontend displays the race and exposes interactive controls.  

## Quickstart
### 1. Install requirements
  ```bash
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

2. Prepare weights
Put trained weights in server/weights/:
    vae.pth  
    mdnrnn.pth  
    controllerA.npz  
    controllerB.npz

3. Start backend
   uvicorn server.run_duel:app --host 0.0.0.0 --port 8000

4. Start UI
   BACKEND_URL=http://localhost:8000 streamlit run app.py

Controls
  Reset — restart duel with given/random seed
  Start / Pause — control simulation
  Per-player knobs
  alpha_action_smoothing
  risk_aversion
  emergency_brake_threshold
  lookahead_horizon
  gating_bias.{rain, drift, straight}
  Backend enforces action clamping and a minimum throttle so cars always move.

API Endpoints
  GET /health — check environment and weights
  POST /control — {action: start|pause|reset}
  POST /update_knobs — update controller parameters
  GET /telemetry — returns step count, rewards, off-track time, and frames

Repo Structure
  server/
  run_duel.py      # FastAPI backend
  models/          # VAE, MDN-RNN, Controller
  weights/         # trained weights go here
configs/
  bounds.yaml      # knob parameter ranges
app.py             # Streamlit UI
requirements.txt
README.md



