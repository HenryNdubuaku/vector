# vector

Programming language for machine learning, built on top of XLA compiler.

## Overview

```python

# vector is designed for machine learning 
n = 64
batch_size = 16
hidden_size = 8
learning_rate = 0.03
train_steps = 30000

# vector has numpy-like vectorized functions 
inputs = reshape(linspace(-pi, pi, n), n, 1)
targets = sin(inputs)
eval_inputs = reshape(linspace(-pi, pi, 9), 9, 1)
eval_targets = sin(eval_inputs)

# vector is functional with modules
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

# export the stableHLO: emits stableHLO text
export(model, "mlp.mlir", eval_inputs)

```

## Install

```sh
cargo install --path . && vector setup
```

Building from source requires `protoc` (`brew install protobuf`). 
`vector setup` downloads the PJRT CPU plugin for your platform into `~/.vector`; on linux, 
`vector setup cuda` (or `rocm`, `oneapi`, `tpu`) adds an accelerator backend, preferred automatically when present. 
`VECTOR_BACKEND=cpu` pins a backend; `PJRT_PLUGIN_PATH` overrides everything. 
TPUs have no f64, so avoid `f64(...)` and f64 `.npy` inputs there.

```sh
vector filename.vec
```

`load` reads `.npy` files (little-endian f32/f64, C order); the tensor becomes a runtime input to the compiled program, so shapes stay static. 
`save(model, "model.safetensors")` writes weights as safetensors with the module structure in the header metadata; `load("model.safetensors")` returns the instance — callable and trainable — as long as its module is defined in the program. PyTorch state_dicts load as plain records (numeric path components like `layers.0` become `layers._0`). Single tensors save to `.npy`. 
`export(model, "model.mlir", example_inputs...)` writes the forward pass as a standalone StableHLO module with the weights baked in as constants (the examples fix the input shapes) — runnable by anything that speaks StableHLO: JAX, IREE, PJRT plugins. 
Output comes only from `print`. Transforms: `grad`, `vmap` (nestable), `jacobian`. 
Loops (`for i in 0..n:`) compile to one XLA while op (and unroll under `grad`, so gradients flow through them); `where(cond, a, b)` selects elementwise with comparisons `< > <= >=`.


## TO-DO:

- plot 
- neuron (trainium) and metal backends