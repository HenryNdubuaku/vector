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

```sh
vector run examples/hello.vec
vector build examples/hello.vec > hello.mlir
```
