# Vector

Programming language for machine learning, built on top of XLA compiler.

## Overview

```python

# vector is designed for machine learning 
n = 64
batch_size = 16
hidden_size = 8
learning_rate = 0.03
train_steps = 30000

# vector ships numpy-like vectorized functions 
inputs = reshape(linspace(-pi, pi, n), n, 1)
targets = sin(inputs)
eval_inputs = reshape(linspace(-pi, pi, 9), 9, 1)
eval_targets = sin(eval_inputs)

# vector is functional like JAX, but with modules
module Mlp(hidden):
  l1 = Linear(1, hidden)
  l2 = Linear(hidden, 1)

  forward(self, x):
    self.l2(tanh(self.l1(x)))

  loss(self, inputs, targets):
    error = self(inputs) - targets
    mean(error * error)

# vector is bult on XLA compiler
model = Mlp(hidden_size)

# vector programs run on Nvidia/AMD/TPUs and more  
for step in 0..train_steps:
  offset = mod(step, n / batch_size) * batch_size
  x = slice(inputs, offset, batch_size)
  t = slice(targets, offset, batch_size)
  model = model - learning_rate * grad(model.loss, x, t)

# vector save weights as safetensors for cross-compatibility
save(model, "mlp.safetensors")
model = load("mlp.safetensors")

# vector is designed to look like python 
print(model.loss(inputs, targets))
print(model(eval_inputs))
print(eval_targets)

# vector saves and loads data as numpy .npy files
save(model(eval_inputs), "predictions.npy")
print(load("predictions.npy") - eval_targets)

# vector reads and writes csv tables as records of columns like pandas
save({x: inputs, sin: targets, mlp: model(inputs)}, "predictions.csv")
table = load("predictions.csv")
print(mean(table.mlp - table.sin))

# vector plots with a matplotlib-like interface, rendered as svg
plot(inputs, targets, "sin")
plot(inputs, model(inputs), "mlp")
title("sin approximation")
savefig("sin.svg")

# export the stableHLO: emits stableHLO text
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

## Roadmap

- pillow 
- neuron (trainium) and metal backends
- test on GPU
- test on TPU 

## Contributing

- Follow the intuitive and minimalist coding established in the codebase.
- Try bringing table, plot, etc up to parity with equivalent Python libs.
- Create an official Docker image.
- Make the docs intuitive.