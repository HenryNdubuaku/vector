# Vector

<img src="assets/banner.png" alt="Logo" style="border-radius: 30px; width: 100%;">

A programming language for machine learning, compiled to CPUs, GPUs and TPUs through [XLA](https://openxla.org/xla).

ML in Python relies on special libraries with C/C++ backend; PyTorch, NumPy, JAX, Pandas. JAX is particularly fast, thanks to the XLA compiler which compiles for CPU, TPUs/Nvidia GPUs, AMD GPUs, Apple GPUs, etc. 

Vector brings JAX-level speed across the entire program with Pythonic-syntax and functional paradigm, your entire code can run on accelerators, not just the training loop. 

- 200 full-batch gradient-descent steps of a 1→1024→1024→1 tanh network on 2048 points of sin(x), f32. 
- JAX runs a jitted `fori_loop`; PyTorch runs both its standard eager loop and a `torch.compile`d step. 
- Timings are the median of 5 runs after one warm-up, excluding compilation. 

| Device                    | Vector    | JAX        | PyTorch (eager) | PyTorch (compiled) |
| ------------------------- | --------- | ---------- | --------------- | ------------------ |
| Apple M5 Max CPU (ARM)    | **1.51s** | 1.62s      | 2.11s           | 2.15s              |
| Apple M5 Max GPU (Metal)  | **0.27s** | —          | 0.32s           | 0.30s              |
| GPU-box CPU (x86)         | 7.60s     | **6.90s**  | 13.50s          | 14.81s             |
| NVIDIA RTX 4000 Ada       | **0.09s** | **0.09s**  | 0.29s           | 0.25s              |
| TPU-VM CPU (x86)          | **1.70s** | 3.46s      | —               | —                  |
| Google TPU                | **0.01s** | **0.01s**  | —               | —                  |

## Overview

The tour below is one program: train a network to fit sin(x):

```python
n = 16000
hidden_size = 1024
learning_rate = 0.03
epochs = 30
batch_size = 32
batches = 500

inputs = reshape(linspace(-pi, pi, n), n, 1)
targets = sin(inputs)
```

Vector modules are analogous to PyTorch nn.Module:

```python
module Mlp(hidden):
  l1 = Linear(1, hidden)
  l2 = Linear(hidden, 1)

  forward(self, x):
    self.l2(tanh(self.l1(x)))

  loss(self, inputs, targets):
    error = self(inputs) - targets
    mean(error * error)

model = Mlp(hidden_size)
```

Training is whole-model arithmetic and the loop compiles to a single XLA op. 
Each epoch shuffles and slices minibatches, like a DataLoader with `shuffle=True`:

```python
function train_epoch(model, inputs, targets, lr, n, batch, batches):
  m = model
  perm = permutation(n)
  xs = take(inputs, perm)
  ts = take(targets, perm)
  for step in 0..batches:
    x = slice(xs, step * batch, batch)
    t = slice(ts, step * batch, batch)
    m = m - lr * grad(m.loss, x, t)
  m

for epoch in 0..epochs:
  model = train_epoch(model, inputs, targets, learning_rate, n, batch_size, batches)
  print(model.loss(inputs, targets))
```

Weights save as safetensors, tensors as numpy `.npy`:

```python
save(model, "mlp.safetensors")
model = load("mlp.safetensors")

eval_inputs = reshape(linspace(-pi, pi, 9), 9, 1)
eval_targets = sin(eval_inputs)
print(model(eval_inputs))
print(eval_targets)

save(model(eval_inputs), "predictions.npy")
print(load("predictions.npy") - eval_targets)
```

The trained forward pass exports as [StableHLO](https://openxla.org/stablehlo):

```python
export(model, "mlp.mlir", eval_inputs)
```

You can serve the exported model over http:
```sh
vector serve mlp.mlir 8080
curl -d '{"inputs": [[[-3.14], [-2.36], [-1.57], [-0.79], [0.0], [0.79], [1.57], [2.36], [3.14]]]}' http://127.0.0.1:8080/
```

A table is a record of columns, saved and loaded as `.csv`, like pandas:

```python
save({x: inputs, sin: targets, mlp: model(inputs)}, "predictions.csv")
table = load("predictions.csv")
print(mean(table.mlp - table.sin))
```

Plots are matplotlib-style, rendered as `.svg`:

```python
plot(inputs, targets, "sin")
plot(inputs, model(inputs), "mlp")
title("sin approximation")
savefig("sin.svg")
```

Image processing capabilites are natively shipped:

```python
grid = sin(linspace(-pi, pi, 64))
surface = 0.5 + 0.5 * matmul(reshape(grid, 64, 1), reshape(grid, 1, 64))
save(resize(surface, 32, 32), "surface.png")
imshow(load("surface.png"))
title("sin(x) * sin(y)")
savefig("surface.svg")
```

Audio is a record `{samples, rate}`:

```python
tone = sin(linspace(0.0, 1382.3, 4000))
save({samples: tone * 0.5, rate: 8000.0}, "tone.wav")
```

## Get Started

**1. Install** on any machine with a CPU, Nvidia GPU, TPU, and AMD GPU:
```sh
curl -fsSL https://raw.githubusercontent.com/HenryNdubuaku/vector/main/install.sh | sh && . "$HOME/.cargo/env"
```

**2. Check the machine**: trains a small model on the CPU and the accelerator; if anything is missing, vector prints the exact commands to fix it:
```sh
vector test
```

**3. Run the tour**: paste the overview cells into a file `filename.vec`:
```sh
vector filename.vec
```

Programs run on the CPU by default; add `--accelerate` to run on the machine's GPU or TPU:
```sh
vector filename.vec --accelerate
```

**4. Read more**: [docs/reference.md](docs/reference.md) covers the whole language; [example project](example/) is a simple ML project.

## Roadmap

| When           | Goal                           |
| -------------- | ------------------------------ |
| July 2026      | Finish syntax and behaviour    |
| August 2026    | Parity with Python ML libs     |
| September 2026 | Large-scale distributed ML     |
| October 2026   | Vector libraries               |
| November 2026  | **Release v1**                 |

## Contributing

- [todo.md](todo.md) is the official list, pick anything from it. 
- The core items are ordered; the ecosystem tracks (notebooks, editor support, packages, docker) are self-contained and make good first contributions.
- Follow the intuitive and minimalist coding established in the codebase.
- `cargo test` must stay green: the golden tests are the merge gate, and the docs coverage test keeps the reference honest.