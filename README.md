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

```vec
-- train.vec
xs = load("xs.npy")

fn norm_sq(v):
  sum(v * v)

print(vmap(norm_sq, xs))
print(grad(norm_sq, f64([3.0, 4.0])))
```

```sh
vector run train.vec
vector build train.vec > train.mlir
```

`load` reads `.npy` files (little-endian f32/f64, C order); the tensor becomes a runtime input to the compiled program, so shapes stay static. Output comes only from `print`. Transforms: `grad`, `vmap` (nestable), `jacobian`.
