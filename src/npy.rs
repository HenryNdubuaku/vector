use std::fs;
use std::io::Read;

use pjrt::HostBuffer;

use crate::die;
use crate::graph::Dtype;

#[derive(Debug, Clone)]
pub struct InputSpec {
    pub path: String,
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

pub fn npy_host_buffer(spec: &InputSpec) -> HostBuffer {
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
    let data = &bytes[offset..offset + count * size];
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
        Dtype::I1 => unreachable!(),
    }
}
