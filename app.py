# ui/app.py
import os, time, requests, base64
import streamlit as st
import yaml

BOUNDS_PATH = os.getenv("BOUNDS_PATH", "configs/bounds.yaml")
API = os.getenv("DUEL_API", "http://localhost:8000")

with open(BOUNDS_PATH, "r") as f:
    BOUNDS = yaml.safe_load(f)

st.set_page_config(page_title="VMC Duel", layout="wide")
st.title("V-M-C World Models Duel (2 Cars Head-to-Head)")

# --- Control bar ---
col0, col1, col2, col3 = st.columns([1,1,1,2])
with col0:
    if st.button("Start", use_container_width=True):
        requests.post(f"{API}/control", json={"action":"start"})
with col1:
    if st.button("Pause", use_container_width=True):
        requests.post(f"{API}/control", json={"action":"pause"})
with col2:
    seed_in = st.number_input("Seed", min_value=0, max_value=10_000, value=123, step=1)
    if st.button("Reset", use_container_width=True):
        requests.post(f"{API}/control", json={"action":"reset","seed":int(seed_in)})
with col3:
    st.markdown("**Checkpoints**: knob updates apply every few seconds for fairness.")

st.markdown("---")

# --- Player panels ---
left, right = st.columns(2)

def knob_slider(label, bounds, key):
    lo, hi = bounds
    val = st.session_state.get(key, (lo+hi)/2)
    return st.slider(label, min_value=float(lo), max_value=float(hi), value=float(val), key=key)

def gating_group(prefix, bounds: dict):
    vals = {}
    for subk, rng in bounds.items():
        key = f"{prefix}_{subk}"
        vals[subk] = st.slider(f"{subk}", float(rng[0]), float(rng[1]), 0.0, key=key)
    return vals

with left:
    st.subheader("Player A Knobs")
    a_alpha = knob_slider("Action smoothing α", BOUNDS["alpha_action_smoothing"], "A_alpha")
    a_risk = knob_slider("Risk aversion ρ", BOUNDS["risk_aversion"], "A_risk")
    a_brake= knob_slider("Emergency brake θ", BOUNDS["emergency_brake_threshold"], "A_brake")
    a_h    = int(knob_slider("Lookahead horizon H", BOUNDS["lookahead_horizon"], "A_h"))
    st.caption("MoE gating bias")
    a_gate = gating_group("A_gate", BOUNDS["gating_bias"])
    if st.button("Apply A (next checkpoint)"):
        payload = {
            "player": "A",
            "params": {
                "alpha_action_smoothing": a_alpha,
                "risk_aversion": a_risk,
                "emergency_brake_threshold": a_brake,
                "lookahead_horizon": a_h,
                "gating_bias": a_gate
            }
        }
        requests.post(f"{API}/update_knobs", json=payload)

with right:
    st.subheader("Player B Knobs")
    b_alpha = knob_slider("Action smoothing α", BOUNDS["alpha_action_smoothing"], "B_alpha")
    b_risk = knob_slider("Risk aversion ρ", BOUNDS["risk_aversion"], "B_risk")
    b_brake= knob_slider("Emergency brake θ", BOUNDS["emergency_brake_threshold"], "B_brake")
    b_h    = int(knob_slider("Lookahead horizon H", BOUNDS["lookahead_horizon"], "B_h"))
    st.caption("MoE gating bias")
    b_gate = gating_group("B_gate", BOUNDS["gating_bias"])
    if st.button("Apply B (next checkpoint)"):
        payload = {
            "player": "B",
            "params": {
                "alpha_action_smoothing": b_alpha,
                "risk_aversion": b_risk,
                "emergency_brake_threshold": b_brake,
                "lookahead_horizon": b_h,
                "gating_bias": b_gate
            }
        }
        requests.post(f"{API}/update_knobs", json=payload)

st.markdown("---")

# --- Live Telemetry / Frames ---
fa_col, fb_col = st.columns(2)
hudA = fa_col.empty()
hudB = fb_col.empty()

imgA = fa_col.empty()
imgB = fb_col.empty()

metrics = st.empty()

# simple polling loop
poll_interval = 0.2  # seconds
while True:
    try:
        T = requests.get(f"{API}/telemetry", timeout=2).json()
        ra = T["playerA"]["reward"]; rb = T["playerB"]["reward"]
        ota = T["playerA"]["offtrack_time"]; otb = T["playerB"]["offtrack_time"]

        hudA.markdown(f"**A** — Reward: `{ra:.1f}` | Off-track: `{ota:.2f}` | Step: `{T['step']}`")
        hudB.markdown(f"**B** — Reward: `{rb:.1f}` | Off-track: `{otb:.2f}` | Seed: `{T['seed']}`")

        if T["playerA"]["frame_b64"]:
            imgA.image(Image.open(io.BytesIO(base64.b64decode(T["playerA"]["frame_b64"]))),
                       caption="Player A", use_column_width=True)
        if T["playerB"]["frame_b64"]:
            imgB.image(Image.open(io.BytesIO(base64.b64decode(T["playerB"]["frame_b64"]))),
                       caption="Player B", use_column_width=True)

        metrics.caption("Running" if T["running"] else "Paused")
    except Exception:
        st.warning("Waiting for server… make sure run_duel.py is running on :8000")

    time.sleep(poll_interval)
