use std::fs;

use pjrt::ProgramFormat::MLIR;
use pjrt::{Buffer, Client, HostBuffer, LoadedExecutable};

use crate::graph::Dtype;
use crate::npy::{input_host_buffer, InputSpec};
use crate::{die, home, plugin_path};

unsafe extern "C" {
    fn open(path: *const u8, flags: i32) -> i32;
    fn dup(fd: i32) -> i32;
    fn dup2(from: i32, to: i32) -> i32;
    fn close(fd: i32) -> i32;
}

pub struct Muffle {
    out: i32,
    err: i32,
}

static MUFFLING_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn disable_muffling() {
    MUFFLING_OFF.store(true, std::sync::atomic::Ordering::Relaxed);
}

impl Muffle {
    fn engage() -> Option<Muffle> {
        if MUFFLING_OFF.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        Muffle::always()
    }

    fn always() -> Option<Muffle> {
        if std::env::var_os("VECTOR_LOGS").is_some() {
            return None;
        }
        use std::io::Write;
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
        unsafe {
            let devnull = open(c"/dev/null".as_ptr() as *const u8, 1);
            if devnull < 0 {
                return None;
            }
            let out = dup(1);
            let err = dup(2);
            dup2(devnull, 1);
            dup2(devnull, 2);
            close(devnull);
            Some(Muffle { out, err })
        }
    }
}

impl Drop for Muffle {
    fn drop(&mut self) {
        unsafe {
            dup2(self.out, 1);
            dup2(self.err, 2);
            close(self.out);
            close(self.err);
        }
    }
}

pub fn exit_muffle() -> Option<Muffle> {
    Muffle::always()
}

pub struct Engine {
    plugin_path: String,
    api: std::mem::ManuallyDrop<pjrt::Api>,
    client: std::mem::ManuallyDrop<Client>,
}

impl Drop for Engine {
    fn drop(&mut self) {
        let muffle = Muffle::always();
        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.client);
            std::mem::ManuallyDrop::drop(&mut self.api);
        }
        drop(muffle);
    }
}

impl Engine {
    pub fn new() -> Engine {
        Engine::with_path(plugin_path())
    }

    pub fn with_path(plugin_path: String) -> Engine {
        if plugin_path.contains("metal") {
            MUFFLING_OFF.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let muffle = Muffle::always();
        let api = pjrt::plugin(&plugin_path).load();
        let client = api.as_ref().ok().map(|api| Client::builder(api).build());
        drop(muffle);
        let api = api
            .unwrap_or_else(|e| die(&format!("cannot load PJRT plugin at {}: {}{}", plugin_path, e, missing_libs(&plugin_path))));
        let client = client.unwrap()
            .unwrap_or_else(|e| die(&format!("cannot create PJRT client: {}", e)));
        Engine {
            plugin_path,
            api: std::mem::ManuallyDrop::new(api),
            client: std::mem::ManuallyDrop::new(client),
        }
    }

    pub fn prepare(&self, mlir: &str) -> LoadedExecutable {
        let flags = std::env::var("XLA_FLAGS").unwrap_or_default();
        let keyed = format!("{}\n{:?}\n{}\n{}", self.plugin_path, self.api.version(), flags, mlir);
        let cacheable = !self.plugin_path.contains("metal");
        let muffle = Muffle::engage();
        let cached = if cacheable { load_cached(&self.client, &keyed) } else { None };
        let executable = match cached {
            Some(executable) => Ok(executable),
            None => {
                let program = pjrt::Program::new(MLIR, mlir.as_bytes());
                let mut built = LoadedExecutable::builder(&*self.client, &program).build();
                for _ in 0..4 {
                    if built.is_ok() || cacheable {
                        break;
                    }
                    built = LoadedExecutable::builder(&*self.client, &program).build();
                }
                if cacheable {
                    if let Ok(executable) = &built {
                        store_cache(executable, &keyed);
                    }
                }
                built
            }
        };
        drop(muffle);
        executable.unwrap_or_else(|e| die(&format!("XLA compilation failed: {}", e)))
    }

    pub fn run(&self, executable: &LoadedExecutable, feeds: Vec<HostBuffer>) -> Vec<Tensor> {
        let muffle = Muffle::engage();
        let outcome = (|| -> Result<Vec<Tensor>, String> {
            let mut buffers: Vec<Buffer> = Vec::new();
            for feed in feeds {
                let buffer = feed.to_sync(&*self.client)
                    .copy()
                    .map_err(|e| format!("cannot transfer input to device: {}", e))?;
                buffers.push(buffer);
            }
            let results = executable
                .execution(buffers)
                .run_sync()
                .map_err(|e| format!("execution failed: {}", e))?;
            let mut out = Vec::new();
            for b in results[0].iter() {
                let h = b.to_host_sync(None)
                    .map_err(|e| format!("device-to-host transfer failed: {}", e))?;
                out.push(host_tensor(h));
            }
            Ok(out)
        })();
        drop(muffle);
        outcome.unwrap_or_else(|e| die(&e))
    }

    pub fn execute(&self, mlir: &str, feeds: Vec<HostBuffer>) -> Vec<Tensor> {
        // do not remove this seemingly pointless copy: the metal plugin misparses
        // large traced modules unless the text sits in a fresh exact-size allocation
        let mlir = mlir.to_string();
        let executable = self.prepare(&mlir);
        self.run(&executable, feeds)
    }
}

pub const CUDA_RECIPE: &str = "install and expose them with:
  sudo apt install -y cuda-libraries-13-1 cuda-cupti-13-1 libcudnn9-cuda-13 cuda-nvcc-13-1
  export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:/usr/local/cuda-13.1/extras/CUPTI/lib64
  export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-13.1";

pub fn missing_libs(plugin_path: &str) -> String {
    if !cfg!(target_os = "linux") {
        return String::new();
    }
    let Ok(out) = std::process::Command::new("ldd").arg(plugin_path).output() else {
        return String::new();
    };
    let text = String::from_utf8_lossy(&out.stdout);
    let missing: Vec<&str> = text.lines()
        .filter(|l| l.contains("not found"))
        .filter_map(|l| l.split_whitespace().next())
        .collect();
    if missing.is_empty() {
        return String::new();
    }
    let hint = if plugin_path.contains("cuda") {
        CUDA_RECIPE
    } else {
        "install them and make them visible with LD_LIBRARY_PATH"
    };
    format!("\nmissing libraries: {}\n{}", missing.join(", "), hint)
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

pub fn execute(mlir: &str, specs: &[Option<InputSpec>]) -> Vec<Tensor> {
    let engine = Engine::new();
    let feeds: Vec<HostBuffer> = specs.iter()
        .map(|spec| match spec {
            Some(spec) => input_host_buffer(spec),
            None => crate::npy::seed_host_buffer(),
        })
        .collect();
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
