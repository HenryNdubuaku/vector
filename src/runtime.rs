use std::fs;

use pjrt::ProgramFormat::MLIR;
use pjrt::{Buffer, Client, HostBuffer, LoadedExecutable};

use crate::graph::Dtype;
use crate::npy::{input_host_buffer, InputSpec};
use crate::{die, home, plugin_path};

pub struct Engine {
    plugin_path: String,
    api: pjrt::Api,
    client: Client,
}

impl Engine {
    pub fn new() -> Engine {
        let plugin_path = plugin_path();
        let api = pjrt::plugin(&plugin_path)
            .load()
            .unwrap_or_else(|e| die(&format!("cannot load PJRT plugin at {}: {}", plugin_path, e)));
        let client = Client::builder(&api)
            .build()
            .unwrap_or_else(|e| die(&format!("cannot create PJRT client: {}", e)));
        Engine { plugin_path, api, client }
    }

    pub fn execute(&self, mlir: &str, feeds: Vec<HostBuffer>) -> Vec<Tensor> {
        let flags = std::env::var("XLA_FLAGS").unwrap_or_default();
        let keyed = format!("{}\n{:?}\n{}\n{}", self.plugin_path, self.api.version(), flags, mlir);
        let executable = load_cached(&self.client, &keyed).unwrap_or_else(|| {
            let program = pjrt::Program::new(MLIR, mlir.as_bytes());
            let executable = LoadedExecutable::builder(&self.client, &program)
                .build()
                .unwrap_or_else(|e| die(&format!("XLA compilation failed: {}", e)));
            store_cache(&executable, &keyed);
            executable
        });
        let buffers: Vec<Buffer> = feeds.into_iter()
            .map(|feed| {
                feed.to_sync(&self.client)
                    .copy()
                    .unwrap_or_else(|e| die(&format!("cannot transfer input to device: {}", e)))
            })
            .collect();
        let results = executable
            .execution(buffers)
            .run_sync()
            .unwrap_or_else(|e| die(&format!("execution failed: {}", e)));
        results[0]
            .iter()
            .map(|b| {
                let h = b.to_host_sync(None)
                    .unwrap_or_else(|e| die(&format!("device-to-host transfer failed: {}", e)));
                host_tensor(h)
            })
            .collect()
    }
}

pub fn fnv64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn cache_paths(keyed: &str) -> (String, String) {
    let hash = fnv64(keyed.as_bytes());
    let dir = format!("{}/.vector/cache", home());
    (format!("{}/{:016x}.mlir", dir, hash), format!("{}/{:016x}.exec", dir, hash))
}

fn load_cached(client: &Client, keyed: &str) -> Option<LoadedExecutable> {
    let (mlir_path, exec_path) = cache_paths(keyed);
    if fs::read_to_string(&mlir_path).ok()? != keyed {
        return None;
    }
    let bytes = fs::read(&exec_path).ok()?;
    client.load_executable(&bytes).ok()
}

fn store_cache(executable: &LoadedExecutable, keyed: &str) {
    let (mlir_path, exec_path) = cache_paths(keyed);
    let Ok(inner) = executable.executable() else { return };
    let Ok(serialized) = inner.serialize() else { return };
    if fs::create_dir_all(format!("{}/.vector/cache", home())).is_err() {
        return;
    }
    if fs::write(&exec_path, serialized.bytes()).is_ok() {
        let _ = fs::write(&mlir_path, keyed);
    }
}

#[derive(Debug, Clone)]
enum TensorData {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

#[derive(Debug, Clone)]
pub struct Tensor {
    data: TensorData,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn graph_dtype(&self) -> Dtype {
        match &self.data {
            TensorData::F32(_) => Dtype::F32,
            TensorData::F64(_) => Dtype::F64,
        }
    }

    pub fn to_host_buffer(&self) -> HostBuffer {
        let dims: Vec<i64> = self.shape.iter().map(|&d| d as i64).collect();
        match &self.data {
            TensorData::F32(v) => HostBuffer::from_data(v.clone(), Some(dims), None),
            TensorData::F64(v) => HostBuffer::from_data(v.clone(), Some(dims), None),
        }
    }

    pub fn le_bytes(&self) -> Vec<u8> {
        match &self.data {
            TensorData::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::F64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }

    pub fn f64_vec(&self) -> Vec<f64> {
        match &self.data {
            TensorData::F32(v) => v.iter().map(|&x| x as f64).collect(),
            TensorData::F64(v) => v.clone(),
        }
    }

    fn dtype(&self) -> &'static str {
        match &self.data {
            TensorData::F32(_) => "f32",
            TensorData::F64(_) => "f64",
        }
    }
}

fn host_tensor(h: HostBuffer) -> Tensor {
    let shape: Vec<usize> = h.dims().iter().map(|&d| d as usize).collect();
    match h {
        HostBuffer::F32(b) => Tensor { data: TensorData::F32(b.data().to_vec()), shape },
        HostBuffer::F64(b) => Tensor { data: TensorData::F64(b.data().to_vec()), shape },
        _ => die("unexpected output dtype from XLA"),
    }
}

pub fn execute(mlir: &str, specs: &[InputSpec]) -> Vec<Tensor> {
    let engine = Engine::new();
    let feeds: Vec<HostBuffer> = specs.iter().map(input_host_buffer).collect();
    engine.execute(mlir, feeds)
}

fn format_typed<T: std::fmt::Display>(data: &[T], shape: &[usize]) -> String {
    fn rec<T: std::fmt::Display>(data: &[T], shape: &[usize]) -> String {
        if shape.is_empty() {
            return format!("{}", &data[0]);
        }
        let inner: usize = shape[1..].iter().product::<usize>().max(1);
        let parts: Vec<String> = (0..shape[0])
            .map(|i| rec(&data[i * inner..(i + 1) * inner], &shape[1..]))
            .collect();
        format!("[{}]", parts.join(", "))
    }
    rec(data, shape)
}

pub fn format_tensor(t: &Tensor) -> String {
    let values = match &t.data {
        TensorData::F32(v) => format_typed(v, &t.shape),
        TensorData::F64(v) => format_typed(v, &t.shape),
    };
    format!("{} : {}", values, t.dtype())
}

impl Tensor {
    pub fn format_row(&self, i: usize) -> String {
        let inner: usize = self.shape[1..].iter().product::<usize>().max(1);
        let values = match &self.data {
            TensorData::F32(v) => format_typed(&v[i * inner..(i + 1) * inner], &self.shape[1..]),
            TensorData::F64(v) => format_typed(&v[i * inner..(i + 1) * inner], &self.shape[1..]),
        };
        format!("{} : {}", values, self.dtype())
    }
}
