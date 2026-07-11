# vector

Programming language for machine learning, compiled to XLA.

Programs are traced to StableHLO and executed through a PJRT plugin — there is no interpreter.

## Setup

Requires `protoc` (`brew install protobuf`) and a PJRT CPU plugin:

```sh
mkdir -p plugins && cd plugins
gh release download -R zml/pjrt-artifacts -p 'pjrt-cpu_darwin-arm64.tar.gz'
tar xzf pjrt-cpu_darwin-arm64.tar.gz && rm pjrt-cpu_darwin-arm64.tar.gz
```

The runtime loads `plugins/libpjrt_cpu.dylib` by default; override with `PJRT_PLUGIN_PATH`.

## Run

```sh
cargo run -- examples/hello.vec
```
