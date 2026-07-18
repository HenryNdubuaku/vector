use std::fs;
use std::io::Read;

use pjrt::HostBuffer;

use crate::die;
use crate::graph::Dtype;
use crate::runtime::Tensor;

#[derive(Debug, Clone)]
pub struct InputSpec {
    pub path: String,
    pub entry: Option<String>,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

pub fn npy_meta(path: &str) -> (Vec<usize>, Dtype, usize) {
    let mut f = fs::File::open(path)
        .unwrap_or_else(|e| die(&format!("cannot open {}: {}", path, e)));
    let mut intro = [0u8; 8];
    f.read_exact(&mut intro)
        .unwrap_or_else(|e| die(&format!("cannot read {}: {}", path, e)));
    if &intro[0..6] != b"\x93NUMPY" {
        die(&format!("{} is not a .npy file", path));
    }
    let header_len = match intro[6] {
        1 => {
            let mut b = [0u8; 2];
            f.read_exact(&mut b).unwrap_or_else(|e| die(&format!("cannot read {}: {}", path, e)));
            u16::from_le_bytes(b) as usize
        }
        2 => {
            let mut b = [0u8; 4];
            f.read_exact(&mut b).unwrap_or_else(|e| die(&format!("cannot read {}: {}", path, e)));
            u32::from_le_bytes(b) as usize
        }
        v => die(&format!("unsupported .npy version {} in {}", v, path)),
    };
    let mut header = vec![0u8; header_len];
    f.read_exact(&mut header)
        .unwrap_or_else(|e| die(&format!("cannot read {}: {}", path, e)));
    let header = String::from_utf8_lossy(&header).to_string();
    let dtype = if header.contains("'<f4'") {
        Dtype::F32
    } else if header.contains("'<f8'") {
        Dtype::F64
    } else {
        die(&format!("unsupported dtype in {} (need little-endian f32/f64): {}", path, header.trim()));
    };
    if !header.contains("'fortran_order': False") {
        die(&format!("{} is fortran-ordered; only C order is supported", path));
    }
    let open = header.find('(')
        .unwrap_or_else(|| die(&format!("malformed .npy header in {}: {}", path, header.trim())));
    let close = header[open..].find(')')
        .unwrap_or_else(|| die(&format!("malformed .npy header in {}: {}", path, header.trim())));
    let shape: Vec<usize> = header[open + 1..open + close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap_or_else(|_| die(&format!("malformed shape in {}: {}", path, header.trim()))))
        .collect();
    let data_offset = 8 + if intro[6] == 1 { 2 } else { 4 } + header_len;
    (shape, dtype, data_offset)
}

pub fn host_buffer(dtype: Dtype, shape: &[usize], data: &[u8]) -> HostBuffer {
    let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    match dtype {
        Dtype::F32 => {
            let vals: Vec<f32> = data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            HostBuffer::from_data(vals, Some(dims), None)
        }
        Dtype::F64 => {
            let vals: Vec<f64> = data.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
            HostBuffer::from_data(vals, Some(dims), None)
        }
        Dtype::I1 | Dtype::I32 | Dtype::I64 | Dtype::U32 => unreachable!(),
    }
}

static IDX: std::sync::Mutex<Vec<(String, std::time::SystemTime, std::sync::Arc<(Vec<usize>, Vec<f32>)>)>> =
    std::sync::Mutex::new(Vec::new());

fn idx_data(path: &str) -> std::sync::Arc<(Vec<usize>, Vec<f32>)> {
    let mtime = fs::metadata(path)
        .and_then(|m| m.modified())
        .unwrap_or_else(|e| die(&format!("cannot open {}: {}", path, e)));
    let mut cache = IDX.lock().unwrap();
    if let Some((_, t, data)) = cache.iter().find(|(p, _, _)| p == path) {
        if *t == mtime {
            return data.clone();
        }
    }
    let raw = fs::read(path)
        .unwrap_or_else(|e| die(&format!("cannot read {}: {}", path, e)));
    let bytes = crate::image::gunzip(&raw, path);
    if bytes.len() < 4 || bytes[0] != 0 || bytes[1] != 0 || bytes[2] != 0x08 {
        die(&format!("{} is not an idx file of bytes (the mnist format)", path));
    }
    let ndims = bytes[3] as usize;
    let header = 4 + 4 * ndims;
    if ndims == 0 || bytes.len() < header {
        die(&format!("{} has a malformed idx header", path));
    }
    let shape: Vec<usize> = (0..ndims)
        .map(|d| u32::from_be_bytes(bytes[4 + 4 * d..8 + 4 * d].try_into().unwrap()) as usize)
        .collect();
    let count: usize = shape.iter().product();
    if bytes.len() != header + count {
        die(&format!("{} is truncated: {} values for shape {:?}", path, bytes.len() - header, shape));
    }
    let vals: Vec<f32> = bytes[header..].iter().map(|&b| b as f32).collect();
    let data = std::sync::Arc::new((shape, vals));
    cache.retain(|(p, _, _)| p != path);
    cache.push((path.to_string(), mtime, data.clone()));
    data
}

pub fn idx_meta(path: &str) -> Vec<usize> {
    idx_data(path).0.clone()
}

pub fn idx_host_buffer(path: &str, shape: &[usize]) -> HostBuffer {
    let data = idx_data(path);
    if data.0 != shape {
        die(&format!("{} changed since compilation: {:?} vs {:?}", path, data.0, shape));
    }
    let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    HostBuffer::from_data(data.1.clone(), Some(dims), None)
}

pub fn seed_host_buffer() -> HostBuffer {
    let seed = match std::env::var("VECTOR_SEED") {
        Ok(s) => s.parse::<u64>()
            .unwrap_or_else(|_| die(&format!("VECTOR_SEED must be a number, got {}", s))),
        Err(_) => std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64 ^ (d.as_secs() << 20) ^ std::process::id() as u64)
            .unwrap_or(0),
    };
    HostBuffer::from_data(vec![(seed & 0xFFFFFF) as f32], Some(vec![]), None)
}

pub fn input_host_buffer(spec: &InputSpec) -> HostBuffer {
    if let Some(name) = &spec.entry {
        if spec.path.ends_with(".csv") {
            return crate::table::csv_host_buffer(&spec.path, name, &spec.shape);
        }
        if spec.path.ends_with(".txt") {
            return crate::text::tokens_host_buffer(&spec.path, name, &spec.shape);
        }
        return crate::safetensors::tensor_host_buffer(&spec.path, name, &spec.shape, spec.dtype);
    }
    if spec.path.ends_with(".png") {
        return crate::image::png_host_buffer(&spec.path, &spec.shape);
    }
    if spec.path.ends_with(".wav") {
        return crate::audio::wav_host_buffer(&spec.path, &spec.shape);
    }
    if spec.path.ends_with(".txt") {
        return crate::text::txt_host_buffer(&spec.path, &spec.shape);
    }
    if spec.path.ends_with(".gz") {
        return idx_host_buffer(&spec.path, &spec.shape);
    }
    let (shape, dtype, offset) = npy_meta(&spec.path);
    if shape != spec.shape || dtype != spec.dtype {
        die(&format!("{} changed since compilation: {:?} {} vs {:?} {}",
                     spec.path, shape, dtype.name(), spec.shape, spec.dtype.name()));
    }
    let bytes = fs::read(&spec.path)
        .unwrap_or_else(|e| die(&format!("cannot read {}: {}", spec.path, e)));
    let count: usize = shape.iter().product();
    let size = if dtype == Dtype::F32 { 4 } else { 8 };
    if bytes.len() < offset + count * size {
        die(&format!("{} is truncated: expected {} data bytes, found {}",
                     spec.path, count * size, bytes.len() - offset));
    }
    host_buffer(dtype, &shape, &bytes[offset..offset + count * size])
}

pub fn write_npy(path: &str, t: &Tensor) {
    let descr = match t.graph_dtype() {
        Dtype::F32 => "<f4",
        Dtype::F64 => "<f8",
        _ => unreachable!("saves are checked at trace time"),
    };
    let dims: Vec<String> = t.shape().iter().map(|d| d.to_string()).collect();
    let shape = match dims.len() {
        0 => "()".to_string(),
        1 => format!("({},)", dims[0]),
        _ => format!("({})", dims.join(", ")),
    };
    let mut header = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}", descr, shape);
    let pad = (64 - (10 + header.len() + 1) % 64) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend((header.len() as u16).to_le_bytes());
    bytes.extend(header.as_bytes());
    bytes.extend(t.le_bytes());
    fs::write(path, bytes).unwrap_or_else(|e| die(&format!("cannot write {}: {}", path, e)));
}
