#!/bin/sh
set -u

STEPS="${BENCH_STEPS:-200}"
export BENCH_STEPS="$STEPS"

echo "machine: $(uname -ms), $STEPS training steps of a 1-1024-1024-1 tanh net on 2048 points"
echo "versions: $(vector version 2>/dev/null || echo 'vector missing'), $(python3 -c 'import sys; print("python", sys.version.split()[0])' 2>/dev/null)"
python3 -c 'import jax; print("jax", jax.__version__)' 2>/dev/null
python3 -c 'import torch; print("torch", torch.__version__)' 2>/dev/null
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1
echo "timings are the median of 5 runs after one warm-up; identical weights everywhere, so losses must match"
echo

OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT

echo "== vector =="
vector benchmark "$STEPS" | tee -a "$OUT" || echo "vector failed"
echo

echo "== python/jax =="
if JAX_OUT="$(python3 - <<'EOF' 2>/dev/null
import os, time, warnings
warnings.filterwarnings("ignore")
import jax, jax.numpy as jnp

n, h, lr = 2048, 1024, 1e-3
steps = int(os.environ["BENCH_STEPS"])

def init():
    return dict(
        w1=jnp.sin(jnp.arange(h, dtype=jnp.float32)).reshape(1, h),
        b1=jnp.zeros(h, jnp.float32),
        w2=jnp.sin(jnp.arange(h * h, dtype=jnp.float32)).reshape(h, h) * 0.03,
        b2=jnp.zeros(h, jnp.float32),
        w3=jnp.sin(jnp.arange(h, dtype=jnp.float32)).reshape(h, 1) * 0.05,
    )

x = jnp.linspace(-jnp.pi, jnp.pi, n, dtype=jnp.float32).reshape(n, 1)
t = jnp.sin(x)

def loss(p):
    h1 = jnp.tanh(x @ p["w1"] + p["b1"])
    h2 = jnp.tanh(h1 @ p["w2"] + p["b2"])
    d = h2 @ p["w3"] - t
    return jnp.mean(d * d)

@jax.jit
def train(p):
    def step(i, p):
        g = jax.grad(loss)(p)
        return jax.tree.map(lambda w, g: w - lr * g, p, g)
    return jax.lax.fori_loop(0, steps, step, p)

devices = {d.platform: d for d in jax.devices("cpu")}
try:
    accel = jax.devices()[0]
    if accel.platform != "cpu":
        devices[accel.platform] = accel
except RuntimeError:
    pass

import statistics
for name, dev in devices.items():
    with jax.default_device(dev):
        p = init()
        before = float(loss(p))
        trained = jax.block_until_ready(train(p))
        times = []
        for _ in range(5):
            p = init()
            t0 = time.perf_counter()
            trained = jax.block_until_ready(train(p))
            times.append(time.perf_counter() - t0)
        print(f"{name}: ok — loss {before:.4f} -> {float(loss(trained)):.4f} (trained in {statistics.median(times):.2f}s median of 5)")
EOF
)"; then
    printf '%s\n' "$JAX_OUT" | tee -a "$OUT"
else
    echo "jax not installed (pip install jax); skipped"
fi
echo

echo "== python/pytorch =="
if TORCH_OUT="$(python3 - <<'EOF' 2>/dev/null
import os, time, math
import torch

n, h, lr = 2048, 1024, 1e-3
steps = int(os.environ["BENCH_STEPS"])

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")
if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    devices.append("mps")

def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

def init(device):
    return dict(
        w1=torch.sin(torch.arange(h, dtype=torch.float32, device=device)).reshape(1, h),
        b1=torch.zeros(h, device=device),
        w2=torch.sin(torch.arange(h * h, dtype=torch.float32, device=device)).reshape(h, h) * 0.03,
        b2=torch.zeros(h, device=device),
        w3=torch.sin(torch.arange(h, dtype=torch.float32, device=device)).reshape(h, 1) * 0.05,
    )

def loss(p, x, t):
    h1 = torch.tanh(x @ p["w1"] + p["b1"])
    h2 = torch.tanh(h1 @ p["w2"] + p["b2"])
    d = h2 @ p["w3"] - t
    return (d * d).mean()

def train(p, x, t):
    for _ in range(steps):
        for v in p.values():
            v.requires_grad_(True)
        l = loss(p, x, t)
        gs = torch.autograd.grad(l, list(p.values()))
        with torch.no_grad():
            p = {k: (v - lr * g).detach() for (k, v), g in zip(p.items(), gs)}
    return p

def train_compiled(p, x, t):
    gfn = torch.func.grad(lambda p: loss(p, x, t))
    step = torch.compile(lambda p: {k: v - lr * g for (k, v), g in zip(p.items(), gfn(p).values())})
    for _ in range(steps):
        p = step(p)
    return p

for device in devices:
    x = torch.linspace(-math.pi, math.pi, n, device=device).reshape(n, 1)
    t = torch.sin(x)
    with torch.no_grad():
        before = float(loss(init(device), x, t))
    import statistics
    for name, impl in [("eager", train), ("compiled", train_compiled)]:
        try:
            impl(init(device), x, t)
            sync(device)
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                trained = impl(init(device), x, t)
                sync(device)
                times.append(time.perf_counter() - t0)
            with torch.no_grad():
                after = float(loss(trained, x, t))
            print(f"{device} ({name}): ok — loss {before:.4f} -> {after:.4f} (trained in {statistics.median(times):.2f}s median of 5)")
        except Exception as e:
            print(f"{device} ({name}): skipped ({type(e).__name__})")
EOF
)"; then
    printf '%s\n' "$TORCH_OUT" | tee -a "$OUT"
else
    echo "pytorch not installed (pip install torch); skipped"
fi
echo

spread=$(grep -o 'loss [0-9.]* -> [0-9.]*' "$OUT" | awk '
    { if (NR == 1 || $2 < bmin) bmin = $2; if (NR == 1 || $2 > bmax) bmax = $2;
      if (NR == 1 || $4 < amin) amin = $4; if (NR == 1 || $4 > amax) amax = $4; }
    END { print (bmax - bmin <= 0.001 && amax - amin <= 0.001) ? "ok" : "bad" }')
if [ "$spread" = "ok" ]; then
    echo "correctness: all frameworks computed the same losses (within 0.001; bf16 devices like TPUs round differently)"
else
    echo "WARNING: losses differ across frameworks by more than 0.001; the comparison is not valid"
    grep -o 'loss [0-9.]* -> [0-9.]*' "$OUT" | sort -u
fi
