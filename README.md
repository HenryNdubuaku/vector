# Vector

Programming language for machine learning, built on top of XLA compiler.

```python

# vector is designed for machine learning 
n = 16000
hidden_size = 1024
learning_rate = 0.03
epochs = 30
batch_size = 32
batches = 500

# vector ships numpy-like vectorized functions 
inputs = reshape(linspace(-pi, pi, n), n, 1)
targets = sin(inputs)
eval_inputs = reshape(linspace(-pi, pi, 9), 9, 1)
eval_targets = sin(eval_inputs)

# interleave the sorted samples so every batch spans the domain
shuffled_x = reshape(transpose(reshape(reshape(inputs, n), batch_size, batches)), n, 1)
shuffled_t = reshape(transpose(reshape(reshape(targets, n), batch_size, batches)), n, 1)

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
fn train_epoch(model, xs, ts, lr, batch, batches):
  m = model
  for step in 0..batches:
    offset = step * batch
    x = slice(xs, offset, batch)
    t = slice(ts, offset, batch)
    m = m - lr * grad(m.loss, x, t)
  m

model = Mlp(hidden_size)

# vector programs run on Nvidia/AMD/TPUs and more;
# print inside a loop logs one neat line per iteration
for epoch in 0..epochs:
  model = train_epoch(model, shuffled_x, shuffled_t, learning_rate, batch_size, batches)
  print(model.loss(inputs, targets))

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

# vector loads, resizes, crops and saves png images as tensors
grid = sin(linspace(-pi, pi, 64))
surface = 0.5 + 0.5 * matmul(reshape(grid, 64, 1), reshape(grid, 1, 64))
save(resize(surface, 32, 32), "surface.png")
imshow(load("surface.png"))
title("sin(x) * sin(y)")
savefig("surface.svg")

# vector reads, writes and plays audio as wav records {samples, rate}
tone = sin(linspace(0.0, 1382.3, 4000))
save({samples: tone * 0.5, rate: 8000.0}, "tone.wav")

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

- test on GPU
- test on TPU 
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