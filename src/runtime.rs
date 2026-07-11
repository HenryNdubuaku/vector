use pjrt::ProgramFormat::MLIR;
use pjrt::{Buffer, Client, HostBuffer, LoadedExecutable};

use crate::die;
use crate::npy::{npy_host_buffer, InputSpec};
use crate::plugin_path;

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
    let plugin_path = plugin_path();
    let api = pjrt::plugin(&plugin_path)
        .load()
        .unwrap_or_else(|e| die(&format!("cannot load PJRT plugin at {}: {}", plugin_path, e)));
    let client = Client::builder(&api)
        .build()
        .unwrap_or_else(|e| die(&format!("cannot create PJRT client: {}", e)));
    let program = pjrt::Program::new(MLIR, mlir.as_bytes());
    let executable = LoadedExecutable::builder(&client, &program)
        .build()
        .unwrap_or_else(|e| die(&format!("XLA compilation failed: {}", e)));
    let buffers: Vec<Buffer> = specs.iter()
        .map(|spec| {
            npy_host_buffer(spec)
                .to_sync(&client)
                .copy()
                .unwrap_or_else(|e| die(&format!("cannot transfer {} to device: {}", spec.path, e)))
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
