# vector

Programming language for machine learning, compiled to XLA.

Programs are traced to StableHLO and executed through a PJRT plugin — there is no interpreter.

## Install

```sh
cargo install --path .
vector setup
```

Building from source requires `protoc` (`brew install protobuf`). `vector setup` downloads the PJRT CPU plugin for your platform into `~/.vector`; set `PJRT_PLUGIN_PATH` to use a different plugin.

## Use

```python
fn loss(w):
  ys = load("ys.npy")
  d = w - ys
  mean(d * d)

w = [0.0, 0.0]
for step in 0..100:
  w = w - 0.1 * grad(loss, w)
print(w)
print(loss(w))
```

```sh
vector run filename.vec
vector build filename.vec > filename.mlir
```

`load` reads `.npy` files (little-endian f32/f64, C order); the tensor becomes a runtime input to the compiled program, so shapes stay static. 
Output comes only from `print`. Transforms: `grad`, `vmap` (nestable), `jacobian`. 
Loops (`for i in 0..n:`) unroll at trace time, so gradients flow through them; `where(cond, a, b)` selects elementwise with comparisons `< > <= >=`.
