# Vector

Programming language for machine learning, built on top of XLA compiler.

## Overview

The tour below is one program: it trains a small network to approximate sin(x), then saves, plots, exports and serves the result. Paste the cells into one `.vec` file in order and run it.

Vector ships numpy-like vectorized functions, and math is elementwise over any shape — `sin` of a matrix is the matrix of sines. Sample 16,000 points, then chop them into 500 batches of 32, one batch per row (the transpose interleaves the sorted samples so every batch spans the whole domain):

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

Vector is functional like JAX, but with modules. A module packs weights and methods together, and an instance is an immutable value — training never mutates it, it builds an updated one:

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

Training is whole-model arithmetic: `grad` returns gradients shaped like the model, so one subtraction updates every weight. `take(bx, step)` picks row `step` — one minibatch. The loop compiles to a single XLA while op, and `print` inside it logs one line per epoch:

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

Weights save as safetensors and tensors as numpy `.npy` — both readable from Python, and PyTorch checkpoints load back the same way. Evaluate the reloaded model on nine fresh points:

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

A table is just a record of columns, saved and loaded as csv, like pandas:

```python
save({x: inputs, sin: targets, mlp: model(inputs)}, "predictions.csv")
table = load("predictions.csv")
print(mean(table.mlp - table.sin))
```

Plotting is matplotlib-style, rendered as svg:

```python
plot(inputs, targets, "sin")
plot(inputs, model(inputs), "mlp")
title("sin approximation")
savefig("sin.svg")
```

An image is a tensor of pixels in 0..1 — load, resize, crop and save png, and show it in a figure:

```python
grid = sin(linspace(-pi, pi, 64))
surface = 0.5 + 0.5 * matmul(reshape(grid, 64, 1), reshape(grid, 1, 64))
save(resize(surface, 32, 32), "surface.png")
imshow(load("surface.png"))
title("sin(x) * sin(y)")
savefig("surface.svg")
```

Audio is a record `{samples, rate}` — synthesize half a second of A4 and save it as wav:

```python
tone = sin(linspace(0.0, 1382.3, 4000))
save({samples: tone * 0.5, rate: 8000.0}, "tone.wav")
```

One line exports the trained forward pass as StableHLO — the portable graph format that JAX, IREE and every XLA runtime consume. `vector serve` (below) answers http requests with it:

```python
export(model, "mlp.mlir", eval_inputs)
```

## Get Started

Step 1: requirements
- most CPU/GPU/TPU device
- [Rust](https://www.rust-lang.org/tools/install)

Step 2: Build from the source
```sh
git clone https://github.com/HenryNdubuaku/vector.git 
cd vector 
cargo install --path . && vector setup 
```

Step 3: Copy the example from the overview into a .vec file and run with
```sh
vector filename.vec
```
Add `--accelerate` to run on the machine's GPU or TPU — vector picks whichever accelerator is installed:
```sh
vector filename.vec --accelerate
```

Step 4: Serve the exported model over http and query it
```sh
vector serve mlp.mlir 8080
```
```sh
curl http://127.0.0.1:8080/    # model signature: {"inputs":["9x1xf32"],"outputs":["9x1xf32"]}
curl -d '{"inputs": [[[-3.14], [-2.36], [-1.57], [-0.79], [0.0], [0.79], [1.57], [2.36], [3.14]]]}' http://127.0.0.1:8080/
```
The server compiles the model once through XLA and answers with `{"outputs": [...]}`; wrong shapes get a loud `{"error": ...}`. 

## Roadmap

- July 2026: Parity with Python libs, integrate into XLA/Python/ML ecosystem. 
- August 2026: Vector notebooks, integrate into academic curriculums. 
- September 2026: Large-scale distributed ML, integrate into enterprises. 
- October 2026: Vector libraries, ecosystem partnerships.
- November 2026: Self-Hosting, workshops & developer events.
- December 2026: Release v1

## Contributing

- Follow the intuitive and minimalist coding established in the codebase.
- Try bringing table, plot, etc up to parity with equivalent Python libs.
- Create an official Docker image.
- Make the docs intuitive.