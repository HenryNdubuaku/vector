# vector

Programming language for machine learning, compiled to XLA.

Programs are traced to StableHLO and executed through a PJRT plugin — there is no interpreter.

## Install

```sh
cargo install --path . && vector setup
```

Building from source requires `protoc` (`brew install protobuf`). `vector setup` downloads the PJRT CPU plugin for your platform into `~/.vector`; on linux, `vector setup cuda` (or `rocm`, `oneapi`, `tpu`) adds an accelerator backend, preferred automatically when present. `VECTOR_BACKEND=cpu` pins a backend; `PJRT_PLUGIN_PATH` overrides everything. TPUs have no f64, so avoid `f64(...)` and f64 `.npy` inputs there.

## Use

```python
n = 64
batch_size = 16
hidden_size = 8
learning_rate = 0.03
train_steps = 30000

inputs = reshape(linspace(-pi, pi, n), n, 1)
targets = sin(inputs)
eval_inputs = reshape(linspace(-pi, pi, 9), 9, 1)
eval_targets = sin(eval_inputs)

module Mlp(hidden):
  l1 = Linear(1, hidden)
  l2 = Linear(hidden, 1)

  forward(self, x):
    self.l2(tanh(self.l1(x)))

  loss(self, inputs, targets):
    error = self(inputs) - targets
    mean(error * error)

model = Mlp(hidden_size)

print(model.loss(inputs, targets))

for step in 0..train_steps:
  offset = mod(step, n / batch_size) * batch_size
  x = slice(inputs, offset, batch_size)
  t = slice(targets, offset, batch_size)
  model = model - learning_rate * grad(model.loss, x, t)

print(model.loss(inputs, targets))
print(model(eval_inputs))
print(eval_targets)

```

```sh
vector filename.vec
```

`load` reads `.npy` files (little-endian f32/f64, C order); the tensor becomes a runtime input to the compiled program, so shapes stay static. 
`save(model, "model.safetensors")` writes weights as safetensors with the module structure in the header metadata; `load("model.safetensors")` returns the instance — callable and trainable — as long as its module is defined in the program. PyTorch state_dicts load as plain records (numeric path components like `layers.0` become `layers._0`). Single tensors save to `.npy`. 
Output comes only from `print`. Transforms: `grad`, `vmap` (nestable), `jacobian`. 
Loops (`for i in 0..n:`) compile to one XLA while op (and unroll under `grad`, so gradients flow through them); `where(cond, a, b)` selects elementwise with comparisons `< > <= >=`.


## TO-DO:

- plot 
- neuron (trainium) and metal backends