# app.py
import os, time, base64, requests
import streamlit as st

# ---------------- Config ----------------
BACKEND_DEFAULT = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="VMC Duel", layout="wide")
st.title("VMC Duel — Two Cars, One Track")

# --------------- Helpers ----------------
def _get(url, timeout=3.0):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _post(url, payload, timeout=5.0):
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _b64_to_bytes(s: str):
    if not s:
        return None
    try:
        return base64.b64decode(s, validate=False)
    except Exception:
        return None

# --------------- Sidebar ----------------
with st.sidebar:
    st.header("Backend")
    backend_url = st.text_input("Backend URL", BACKEND_DEFAULT, help="e.g., http://localhost:8000").rstrip("/")
    col = st.columns(3)
    with col[0]:
        if st.button("Health"):
            try:
                st.write(_get(f"{backend_url}/health"))
            except Exception as e:
                st.error(f"/health failed: {e}")
    with col[1]:
        if st.button("Reset"):
            try:
                st.write(_post(f"{backend_url}/control", {"action": "reset"}))
            except Exception as e:
                st.error(f"/control reset failed: {e}")
    with col[2]:
        if st.button("Start"):
            try:
                st.write(_post(f"{backend_url}/control", {"action": "start"}))
            except Exception as e:
                st.error(f"/control start failed: {e}")

    st.divider()
    st.subheader("Auto-refresh")
    poll_hz = st.slider("Telemetry refresh (Hz)", 1, 15, 5, 1)
    quality_hint = st.selectbox("Frame decode", ["auto", "jpeg"], index=0,
                                help="auto = let Streamlit detect; jpeg = treat as JPEG bytes explicitly")

# --------------- Knobs ----------------
def knob_panel(player_key: str):
    st.subheader(f"Player {player_key} knobs")
    alpha = st.slider(f"[{player_key}] alpha_action_smoothing", 0.0, 1.0, 0.2, 0.01)
    risk  = st.slider(f"[{player_key}] risk_aversion", -1.0, 1.0, 0.0, 0.01)
    ebrk  = st.slider(f"[{player_key}] emergency_brake_threshold", 0.0, 1.5, 0.8, 0.01)
    look  = st.slider(f"[{player_key}] lookahead_horizon", 1, 30, 10, 1)
    st.caption("MoE gating bias")
    rain  = st.slider(f"[{player_key}] rain", -2.0, 2.0, 0.0, 0.01)
    drift = st.slider(f"[{player_key}] drift", -2.0, 2.0, 0.0, 0.01)
    straight = st.slider(f"[{player_key}] straight", -2.0, 2.0, 0.0, 0.01)

    if st.button(f"Apply {player_key} (next checkpoint)"):
        try:
            payload = {
                "player": player_key,
                "params": {
                    "alpha_action_smoothing": alpha,
                    "risk_aversion": risk,
                    "emergency_brake_threshold": ebrk,
                    "lookahead_horizon": look,
                    "gating_bias": {"rain": rain, "drift": drift, "straight": straight},
                },
            }
            st.success(_post(f"{backend_url}/update_knobs", payload))
        except Exception as e:
            st.error(f"Apply failed: {e}")

colA, colB = st.columns(2)
with colA: knob_panel("A")
with colB: knob_panel("B")

st.divider()

# --------------- Live Telemetry & Frames ----------------
left, right = st.columns(2)
imgA = left.empty()
imgB = right.empty()
stat = st.empty()

period = 1.0 / max(1, poll_hz)

while True:
    try:
        j = _get(f"{backend_url}/telemetry", timeout=3.0)

        running = j.get("running", False)
        step    = j.get("step", 0)
        seed    = j.get("seed", -1)

        A = j.get("playerA", {})
        B = j.get("playerB", {})

        # Decode frames
        A_bytes = _b64_to_bytes(A.get("frame_b64", ""))
        B_bytes = _b64_to_bytes(B.get("frame_b64", ""))

        # Render
        capA = f"A — Reward: {A.get('reward',0):.1f} | Off-track: {A.get('offtrack_time',0):.2f}"
        capB = f"B — Reward: {B.get('reward',0):.1f} | Off-track: {B.get('offtrack_time',0):.2f}"

        if A_bytes:
            if quality_hint == "jpeg":
                imgA.image(A_bytes, caption=capA)
            else:
                imgA.image(A_bytes, caption=capA)
        else:
            imgA.warning("Waiting for A frame…")

        if B_bytes:
            if quality_hint == "jpeg":
                imgB.image(B_bytes, caption=capB)
            else:
                imgB.image(B_bytes, caption=capB)
        else:
            imgB.warning("Waiting for B frame…")

        stat.info(f"running={running} | step={step} | seed={seed}")
    except Exception as e:
        stat.error(f"Fetch failed: {e}")

    time.sleep(period)
