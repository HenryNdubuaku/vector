# Vector

A programming language for machine learning, compiled to CPUs, GPUs and TPUs through [XLA](https://openxla.org/xla).

Why Vector exists:

- **Python** interprets your program and hands fragments to compiled libraries (PyTorch/JAX), in Vector the whole program is the graph: `grad`, `vmap` and modules are language features.
- **Dex** proved differentiable array programming in research. Vector packs data, plots, images, audio, checkpoints and serving in one binary that speaks the ecosystem's formats.
- **Bend** built a novel GPU runtime. Vector bets on XLA, the compiler that already runs the world's ML, on every backend it supports.
- **Mojo** is a Python superset carrying all of Python's surface area. Vector is small on purpose: learnable in an afternoon.

## Overview

The tour below is one program: train a network to fit sin(x), then save, plot, export and serve it. Paste the cells into one file, in order.

Vector's functions are numpy-like, and math is elementwise over any shape — `sin` of a matrix is the matrix of sines. 
Each row of `batches_x` is one batch of 32; the transpose interleaves the sorted samples so every batch spans the domain:

```python
n = 16000
hidden_size = 1024
learning_rate = 0.03
epochs = 30
batch_size = 32
batches = 500

xs = linspace(-pi, pi, n)
inputs = reshape(xs, n, 1)
targets = sin(inputs)

batches_x = transpose(reshape(xs, batch_size, batches))
batches_t = sin(batches_x)
```

Vector is functional like JAX, but with modules. A module packs weights and methods; an instance is a value — training never mutates it, it builds an updated one:

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

Training is whole-model arithmetic: `grad` returns a gradient shaped like the model, so one subtraction updates every weight. 
`take` picks one batch, the loop compiles to a single XLA op, and `print` logs each epoch:

```python
fn train_epoch(model, bx, bt, lr, batch, batches):
  m = model
  for step in 0..batches:
    x = reshape(take(bx, step), batch, 1)
    t = reshape(take(bt, step), batch, 1)
    m = m - lr * grad(m.loss, x, t)
  m

for epoch in 0..epochs:
  model = train_epoch(model, batches_x, batches_t, learning_rate, batch_size, batches)
  print(model.loss(inputs, targets))
```

Weights save as safetensors, tensors as numpy `.npy` — Python reads both, and PyTorch checkpoints load back:

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

An image is a tensor of pixels in 0..1 — load, resize, crop, save `.png`, show:

```python
grid = sin(linspace(-pi, pi, 64))
surface = 0.5 + 0.5 * matmul(reshape(grid, 64, 1), reshape(grid, 1, 64))
save(resize(surface, 32, 32), "surface.png")
imshow(load("surface.png"))
title("sin(x) * sin(y)")
savefig("surface.svg")
```

Audio is a record `{samples, rate}` — here half a second of A4, saved as `.wav`:

```python
tone = sin(linspace(0.0, 1382.3, 4000))
save({samples: tone * 0.5, rate: 8000.0}, "tone.wav")
```

One line exports the trained forward pass as [StableHLO](https://openxla.org/stablehlo), the portable graph format that JAX, IREE and every XLA runtime consume:

```python
export(model, "mlp.mlir", eval_inputs)
```

Serve the exported model over http:
```sh
vector serve mlp.mlir 8080
```

Query it:
```sh
curl -d '{"inputs": [[[-3.14], [-2.36], [-1.57], [-0.79], [0.0], [0.79], [1.57], [2.36], [3.14]]]}' http://127.0.0.1:8080/
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

## Benchmarks

- 200 full-batch gradient-descent steps of a 1→1024→1024→1 tanh network on 2048 points of sin(x), f32. 
- Full-batch steps minimize Python dispatch overhead, which is generous to eager PyTorch. 
- Every framework starts from identical weights; the script verifies all frameworks compute the same losses (0.3586 → 0.0133, within 0.001 — TPUs round f32 matmuls through bf16) and prints the verdict. 
- JAX runs a jitted `fori_loop`; PyTorch runs both its standard eager loop and a `torch.compile`d step. 
- Timings are the median of 5 runs after one warm-up, excluding compilation; the script prints all framework versions and GPU info. 

| Device                    | Vector    | JAX        | PyTorch (eager) | PyTorch (compiled) |
| ------------------------- | --------- | ---------- | --------------- | ------------------ |
| Apple M5 Max CPU          | **1.51s** | 1.62s      | 2.11s           | 2.15s              |
| Apple M5 Max GPU (Metal)  | **0.27s** | —          | 0.32s           | 0.30s              |
| GPU-box CPU (x86)         | 7.60s     | **6.90s**  | 13.50s          | 14.81s             |
| NVIDIA RTX 4000 Ada       | **0.09s** | **0.09s**  | 0.29s           | 0.25s              |
| TPU-VM CPU (x86)          | **3.67s** | —          | —               | —                  |
| Google TPU                | **0.01s** | —          | —               | —                  |

## Roadmap

| When           | Focus                          | Goal                                        |
| -------------- | ------------------------------ | ------------------------------------------- |
| July 2026      | Parity with Python libs        | Integrate into XLA/Python/ML ecosystem      |
| August 2026    | Vector notebooks               | Integrate into academic curriculums         |
| September 2026 | Large-scale distributed ML     | Integrate into enterprises                  |
| October 2026   | Vector libraries               | Ecosystem partnerships                      |
| November 2026  | **Release v1**                 | Workshops & developer events                |
| December 2026  | Self-host & broaden            |                                             |

## Contributing

- Follow the intuitive and minimalist coding established in the codebase.
- Try bringing table, plot, etc up to parity with equivalent Python libs.
- Create an official Docker image, test on different cloud platforms. 
- Make the docs intuitive.