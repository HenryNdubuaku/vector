use std::fs;

use pjrt::HostBuffer;

use crate::die;
use crate::graph::{BVal, Dtype, InputSource, OpKind, TVal};
use crate::runtime::Tensor;
use crate::safetensors::SaveSpec;
use crate::trace::Tracer;

struct Wav {
    format: u16,
    channels: usize,
    rate: u32,
    bits: usize,
    data_start: usize,
    data_len: usize,
}

fn u16le(bytes: &[u8], pos: usize) -> u16 {
    u16::from_le_bytes([bytes[pos], bytes[pos + 1]])
}

fn u32le(bytes: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap())
}

fn parse_wav(bytes: &[u8], path: &str) -> Wav {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        die(&format!("{} is not a .wav file", path));
    }
    let mut fmt = None;
    let mut data = None;
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32le(bytes, pos + 4) as usize;
        let body = pos + 8;
        if body + size > bytes.len() {
            die(&format!("{} is truncated", path));
        }
        match id {
            b"fmt " if size >= 16 => {
                fmt = Some((
                    u16le(bytes, body),
                    u16le(bytes, body + 2) as usize,
                    u32le(bytes, body + 4),
                    u16le(bytes, body + 14) as usize,
                ));
            }
            b"data" => data = Some((body, size)),
            _ => {}
        }
        pos = body + size + (size & 1);
    }
    let Some((format, channels, rate, bits)) = fmt else {
        die(&format!("{} is missing its fmt chunk", path));
    };
    let Some((data_start, data_len)) = data else {
        die(&format!("{} is missing its data chunk", path));
    };
    if !matches!((format, bits), (1, 8 | 16 | 24 | 32) | (3, 32)) {
        die(&format!("{} has an unsupported encoding (format {}, {} bits); re-save as pcm", path, format, bits));
    }
    if channels == 0 || rate == 0 {
        die(&format!("{} has a malformed fmt chunk", path));
    }
    Wav { format, channels, rate, bits, data_start, data_len }
}

fn wav_shape(w: &Wav, path: &str) -> Vec<usize> {
    let frame = w.channels * w.bits / 8;
    let n = w.data_len / frame;
    if n == 0 {
        die(&format!("{} has no samples", path));
    }
    if w.channels == 1 { vec![n] } else { vec![n, w.channels] }
}

pub fn wav_meta(path: &str) -> (Vec<usize>, u32) {
    let bytes = fs::read(path)
        .unwrap_or_else(|e| die(&format!("cannot read {}: {}", path, e)));
    let w = parse_wav(&bytes, path);
    (wav_shape(&w, path), w.rate)
}

pub fn decode_wav(path: &str) -> (Vec<usize>, Vec<f32>) {
    let bytes = fs::read(path)
        .unwrap_or_else(|e| die(&format!("cannot read {}: {}", path, e)));
    let w = parse_wav(&bytes, path);
    let shape = wav_shape(&w, path);
    let count: usize = shape.iter().product();
    let sample = w.bits / 8;
    let mut vals = Vec::with_capacity(count);
    for i in 0..count {
        let o = w.data_start + i * sample;
        let v = match (w.format, w.bits) {
            (1, 8) => (bytes[o] as f32 - 128.0) / 128.0,
            (1, 16) => i16::from_le_bytes([bytes[o], bytes[o + 1]]) as f32 / 32768.0,
            (1, 24) => {
                let raw = (bytes[o] as i32) | ((bytes[o + 1] as i32) << 8) | ((bytes[o + 2] as i32) << 16);
                ((raw << 8) >> 8) as f32 / 8388608.0
            }
            (1, 32) => i32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()) as f32 / 2147483648.0,
            _ => f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()),
        };
        vals.push(v);
    }
    (shape, vals)
}

pub fn wav_host_buffer(path: &str, shape: &[usize]) -> HostBuffer {
    let (found, vals) = decode_wav(path);
    if shape != found {
        die(&format!("{} changed since compilation: {:?} vs {:?}", path, found, shape));
    }
    let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    HostBuffer::from_data(vals, Some(dims), None)
}

pub fn wav_save_spec(tracer: &mut Tracer, v: &TVal, path: &str) -> SaveSpec {
    let expect = "save to .wav expects a record {samples, rate}";
    let TVal::Record(_, fields) = v else { die(expect) };
    let samples = fields.iter().find(|(k, _)| k == "samples").map(|(_, f)| f)
        .unwrap_or_else(|| die(expect));
    let rate = fields.iter().find(|(k, _)| k == "rate").map(|(_, f)| f)
        .unwrap_or_else(|| die(expect));
    let s = match samples {
        TVal::Tensor(b) => b.clone(),
        TVal::Record(..) => die(expect),
    };
    let r = match rate {
        TVal::Tensor(b) => b.clone(),
        TVal::Record(..) => die(expect),
    };
    if s.bdims != 0 || r.bdims != 0 {
        die("save to .wav inside vmap isn't supported");
    }
    if s.val.dtype == Dtype::I1 || r.val.dtype == Dtype::I1 {
        die("cannot save booleans; use where to select values");
    }
    if !matches!(s.val.shape.len(), 1 | 2) || s.val.shape.iter().product::<usize>() == 0 {
        die(&format!("wav samples must be [n] or [n, channels], got {:?}", s.val.shape));
    }
    if s.val.shape.len() == 2 && s.val.shape[1] > 64 {
        die(&format!("wav supports at most 64 channels, got {}", s.val.shape[1]));
    }
    if !r.val.shape.is_empty() {
        die(&format!("wav rate must be a scalar, got shape {:?}", r.val.shape));
    }
    let sf = tracer.convert(&s.val, Dtype::F32);
    let rf = tracer.convert(&r.val, Dtype::F32);
    SaveSpec {
        path: path.to_string(),
        names: vec!["samples".to_string(), "rate".to_string()],
        vals: vec![s.val.clone(), r.val.clone()],
        metadata: Vec::new(),
        value: TVal::Record(None, vec![
            ("samples".to_string(), TVal::Tensor(BVal { val: sf, bdims: 0 })),
            ("rate".to_string(), TVal::Tensor(BVal { val: rf, bdims: 0 })),
        ]),
    }
}

pub fn write_wav(spec: &SaveSpec, tensors: &[Tensor]) {
    let samples = &tensors[0];
    let rate = tensors[1].f64_vec()[0];
    if !rate.is_finite() || rate < 1.0 {
        die(&format!("wav rate must be a positive number, got {}", rate));
    }
    let rate = rate.round() as u32;
    let shape = samples.shape();
    let channels = if shape.len() == 2 { shape[1] } else { 1 };
    let mut data = Vec::new();
    for v in samples.f64_vec() {
        let s = (v.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        data.extend(s.to_le_bytes());
    }
    let mut out = Vec::with_capacity(44 + data.len());
    out.extend(b"RIFF");
    out.extend((36 + data.len() as u32).to_le_bytes());
    out.extend(b"WAVE");
    out.extend(b"fmt ");
    out.extend(16u32.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend((channels as u16).to_le_bytes());
    out.extend(rate.to_le_bytes());
    out.extend((rate * channels as u32 * 2).to_le_bytes());
    out.extend((channels as u16 * 2).to_le_bytes());
    out.extend(16u16.to_le_bytes());
    out.extend(b"data");
    out.extend((data.len() as u32).to_le_bytes());
    out.extend(data);
    fs::write(&spec.path, out)
        .unwrap_or_else(|e| die(&format!("cannot write {}: {}", spec.path, e)));
}

impl Tracer {
    pub fn load_wav(&mut self, path: &str) -> TVal {
        let (shape, rate) = wav_meta(path);
        let existing = self.inputs.iter().find(|(src, _)| {
            matches!(src, InputSource::Audio(p) if p == path)
        }).map(|&(_, id)| id);
        let val = match existing {
            Some(id) => self.val(id),
            None => {
                let val = self.emit(OpKind::Input, vec![], shape, Dtype::F32);
                self.inputs.push((InputSource::Audio(path.to_string()), val.id));
                val
            }
        };
        let rate_val = self.constant(rate as f64, Dtype::F32);
        TVal::Record(None, vec![
            ("samples".to_string(), TVal::Tensor(BVal { val, bdims: 0 })),
            ("rate".to_string(), TVal::Tensor(BVal { val: rate_val, bdims: 0 })),
        ])
    }

    pub fn plan_play(&mut self, v: &TVal) {
        if self.region_depth > 0 {
            die("play inside a for loop isn't supported (loops compile to one XLA while op); play after the loop");
        }
        let path = std::env::temp_dir()
            .join(format!("vector_play_{}.wav", self.plays.len()))
            .to_string_lossy()
            .into_owned();
        let spec = wav_save_spec(self, v, &path);
        self.plays.push(spec);
    }
}
