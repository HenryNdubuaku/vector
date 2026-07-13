# Vector

A programming language for machine learning, compiled to CPUs, GPUs and TPUs through [XLA](https://openxla.org/xla).

Why Vector exists:

- **Python** runs on an interpreter and hands fragments to compiled libraries, JAX and PyTorch trace pieces of your program, and their rules (pure functions, no mutation, static shapes) fight the language around them. In Vector the whole program is the computation graph: no interpreter, no two-language split, and `grad`, `vmap` and modules are language features rather than libraries.
- **Dex** proved elegant differentiable array programming as a research project on top of XLA; Vector aims to be the product version, tables, plots, images, audio, checkpoints and serving in a single binary that speaks the ecosystem's formats (safetensors, numpy, StableHLO).
- **Bend** parallelizes arbitrary recursion on a novel GPU runtime; Vector deliberately compiles to XLA, the compiler that already runs the world's ML, so every program inherits a decade of tensor optimization and every backend XLA supports.
- **Mojo** is a Python superset for systems programming, carrying all of Python's surface area; Vector is small on purpose, a functional core purpose-built for training and inference, learnable in an afternoon.

## Overview

The tour below is one program: it trains a small network to approximate sin(x), then saves, plots, exports and serves the result. 
Sample 16,000 points, then chop them into 500 batches of 32, one batch per row (the transpose interleaves the sorted samples so every batch spans the whole domain).
Vector ships numpy-like vectorized functions, and math is elementwise over any shape.

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

Vector is functional like JAX, but with modules. 
A module packs weights and methods together, and an instance is an immutable value. 
Training never mutates it, it builds an updated one:

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

Training is whole-model arithmetic: `grad` returns gradients shaped like the model, so one subtraction updates every weight. 
`take(bx, step)` picks row `step` — one minibatch. 
The loop compiles to a single XLA while op, and `print` inside it logs one line per epoch:

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

Weights save as safetensors and tensors as numpy `.npy`: both readable from Python, and PyTorch checkpoints load back the same way. Evaluate the reloaded model on nine fresh points:

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

A table is just a record of columns, saved and loaded as `.csv`, like pandas:

```python
save({x: inputs, sin: targets, mlp: model(inputs)}, "predictions.csv")
table = load("predictions.csv")
print(mean(table.mlp - table.sin))
```

Plotting is matplotlib-style, rendered as `.svg`:

```python
plot(inputs, targets, "sin")
plot(inputs, model(inputs), "mlp")
title("sin approximation")
savefig("sin.svg")
```

An image is a tensor of pixels in 0..1: load, resize, crop and save `.png`, and show it in a figure:

```python
grid = sin(linspace(-pi, pi, 64))
surface = 0.5 + 0.5 * matmul(reshape(grid, 64, 1), reshape(grid, 1, 64))
save(resize(surface, 32, 32), "surface.png")
imshow(load("surface.png"))
title("sin(x) * sin(y)")
savefig("surface.svg")
```

Audio is a record `{samples, rate}`: synthesize half a second of A4 and save it as `.wav`:

```python
tone = sin(linspace(0.0, 1382.3, 4000))
save({samples: tone * 0.5, rate: 8000.0}, "tone.wav")
```

One line exports the trained forward pass as [StableHLO](https://openxla.org/stablehlo) — the portable graph format that JAX, IREE and every XLA runtime consume. `vector serve` (below) answers http requests with it:

```python
export(model, "mlp.mlir", eval_inputs)
```

## Get Started

**1. Requirements**: any machine with a CPU, GPU or TPU, plus:
- [Rust](https://www.rust-lang.org/tools/install) (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && . "$HOME/.cargo/env"`)
- libclang and [protoc](https://protobuf.dev/installation/) ≥ 3.15 — on macOS just `brew install protobuf`; on Ubuntu apt's protoc is too old, so:
```sh
apt update && apt install -y libclang-dev unzip
curl -LO https://github.com/protocolbuffers/protobuf/releases/download/v25.3/protoc-25.3-linux-x86_64.zip
unzip -o protoc-25.3-linux-x86_64.zip -d /usr/local bin/protoc 'include/*'
```

**2. Build from source**:`vector setup` detects the machine and installs the right backends:
```sh
git clone https://github.com/HenryNdubuaku/vector.git 
cd vector 
cargo install --path . && vector setup 
```
On NVIDIA machines the cuda backend needs the CUDA 13 runtime, cuDNN 9 and nvcc,`vector setup` warns and names anything missing. On a bare container:
```sh
apt install -y cuda-libraries-13-1 cuda-cupti-13-1 libcudnn9-cuda-13 cuda-nvcc-13-1
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:/usr/local/cuda-13.1/extras/CUPTI/lib64
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-13.1
```

**3. Run the tour**: paste the overview cells into `sin.vec`:
```sh
vector examples/train.vec
```
You should see the loss fall as it trains, then the predictions land on sin(x):
```
epoch 0: 0.19202535 : f32
epoch 1: 0.17512359 : f32
...
epoch 29: 0.00006974556 : f32
[[0.0061098086], [-0.7051838], [-0.99085563], ...] : f32
[[0.00000008742278], [-0.70710677], [-1], ...] : f32
```
Programs run on the CPU by default; add `--accelerate` to run on the machine's GPU or TPU — vector picks whichever accelerator is installed:
```sh
vector examples/train.vec --accelerate
```

**4. Serve the exported model** over http and query it:
```sh
vector serve mlp.mlir 8080
```
```sh
curl http://127.0.0.1:8080/
# {"inputs":["9x1xf32"],"outputs":["9x1xf32"]}

curl -d '{"inputs": [[[-3.14], [-2.36], [-1.57], [-0.79], [0.0], [0.79], [1.57], [2.36], [3.14]]]}' http://127.0.0.1:8080/
# {"outputs":[[[-0.12487758],[-0.6564874],[-0.95240515], ...]]}
```
The server compiles the model once through XLA; wrong shapes get a loud `{"error": ...}`.

**5. Read more**: [docs/reference.md](docs/reference.md) covers the whole language; [docs/examples.md](docs/examples.md) shows every feature as a runnable program with its verified output — generated from the test suite, so it can never go stale.

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