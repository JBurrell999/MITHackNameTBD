import numpy as np

class LinearController:
    def __init__(self, in_dim, out_dim, W=None):
        self.in_dim = in_dim; self.out_dim = out_dim
        if W is None:
            self.W = np.random.randn(out_dim, in_dim) * 0.05
            self.b = np.zeros((out_dim,))
        else:
            self.W, self.b = W["W"], W["b"]
        self.a_prev = np.array([0.0, 0.1, 0.0])  # Small initial gas to get moving

    def act(self, z, h, knobs):
        x = np.concatenate([z, h])           # [in_dim]
        a = self.W @ x + self.b              # linear
        # tanh squash to [-1,1], clamp gas/brake
        a = np.tanh(a)
        # smoothing
        alpha = float(knobs.get("alpha_action_smoothing", 0.2))
        a = (1 - alpha) * a + alpha * self.a_prev
        self.a_prev = a
        # emergency brake (simple)
        thr = float(knobs.get("emergency_brake_threshold", 1.0))
        if np.linalg.norm(h) > thr:     # naive risk proxy
            a[2] = max(a[2], 0.8)            # brake channel
        return np.clip(a, [-1,0,0], [1,1,1]) # steer [-1,1], gas/brake [0,1]

    def get_params(self): return {"W": self.W, "b": self.b}
    def set_params(self, P): self.W, self.b = P["W"], P["b"]
