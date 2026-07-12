use std::fs;
use std::io::Read;

use pjrt::HostBuffer;

use crate::die;
use crate::graph::{BVal, Dtype, InputSource, OpKind, TVal};
use crate::runtime::Tensor;
use crate::trace::Tracer;

pub fn crc32(bytes: &[u8]) -> u32 {
    let mut c = !0u32;
    for &b in bytes {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB88320 ^ (c >> 1) } else { c >> 1 };
        }
    }
    !c
}

fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &x in bytes {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

pub fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { ALPHABET[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { ALPHABET[n as usize & 63] as char } else { '=' });
    }
    out
}

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    bit: u32,
    path: &'a str,
}

impl<'a> Bits<'a> {
    fn bits(&mut self, n: u32) -> u32 {
        let mut v = 0u32;
        for i in 0..n {
            if self.pos >= self.data.len() {
                die(&format!("{} is truncated", self.path));
            }
            v |= (((self.data[self.pos] >> self.bit) & 1) as u32) << i;
            self.bit += 1;
            if self.bit == 8 {
                self.bit = 0;
                self.pos += 1;
            }
        }
        v
    }

    fn align(&mut self) {
        if self.bit != 0 {
            self.bit = 0;
            self.pos += 1;
        }
    }
}

struct Huffman {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

fn huffman(lengths: &[u8]) -> Huffman {
    let mut counts = [0u16; 16];
    for &l in lengths {
        counts[l as usize] += 1;
    }
    counts[0] = 0;
    let mut offsets = [0u16; 16];
    for l in 1..16 {
        offsets[l] = offsets[l - 1] + counts[l - 1];
    }
    let mut symbols = vec![0u16; offsets[15] as usize + counts[15] as usize];
    for (sym, &l) in lengths.iter().enumerate() {
        if l > 0 {
            symbols[offsets[l as usize] as usize] = sym as u16;
            offsets[l as usize] += 1;
        }
    }
    Huffman { counts, symbols }
}

impl Huffman {
    fn decode(&self, b: &mut Bits) -> u16 {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= b.bits(1) as i32;
            let count = self.counts[len] as i32;
            if code - first < count {
                return self.symbols[(index + code - first) as usize];
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        die(&format!("{} has an invalid huffman code", b.path));
    }
}

const LEN_BASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA: [u32; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE: [u16; 30] = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA: [u32; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

fn inflate_block(b: &mut Bits, lit: &Huffman, dist: &Huffman, out: &mut Vec<u8>) {
    loop {
        let sym = lit.decode(b) as usize;
        if sym < 256 {
            out.push(sym as u8);
        } else if sym == 256 {
            return;
        } else {
            let idx = sym - 257;
            if idx >= 29 {
                die(&format!("{} has an invalid length code", b.path));
            }
            let len = LEN_BASE[idx] as usize + b.bits(LEN_EXTRA[idx]) as usize;
            let dsym = dist.decode(b) as usize;
            if dsym >= 30 {
                die(&format!("{} has an invalid distance code", b.path));
            }
            let d = DIST_BASE[dsym] as usize + b.bits(DIST_EXTRA[dsym]) as usize;
            if d > out.len() {
                die(&format!("{} has an invalid back-reference", b.path));
            }
            let start = out.len() - d;
            for k in 0..len {
                let byte = out[start + k];
                out.push(byte);
            }
        }
    }
}

fn fixed_tables() -> (Huffman, Huffman) {
    let mut lit = [0u8; 288];
    for (i, l) in lit.iter_mut().enumerate() {
        *l = match i {
            0..=143 => 8,
            144..=255 => 9,
            256..=279 => 7,
            _ => 8,
        };
    }
    (huffman(&lit), huffman(&[5u8; 30]))
}

fn dynamic_tables(b: &mut Bits) -> (Huffman, Huffman) {
    let hlit = b.bits(5) as usize + 257;
    let hdist = b.bits(5) as usize + 1;
    let hclen = b.bits(4) as usize + 4;
    const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
    let mut cl_lengths = [0u8; 19];
    for &slot in ORDER.iter().take(hclen) {
        cl_lengths[slot] = b.bits(3) as u8;
    }
    let cl = huffman(&cl_lengths);
    let mut lengths: Vec<u8> = Vec::with_capacity(hlit + hdist);
    while lengths.len() < hlit + hdist {
        let sym = cl.decode(b);
        match sym {
            0..=15 => lengths.push(sym as u8),
            16 => {
                let prev = *lengths.last()
                    .unwrap_or_else(|| die(&format!("{} has an invalid code-length repeat", b.path)));
                for _ in 0..3 + b.bits(2) {
                    lengths.push(prev);
                }
            }
            17 => {
                for _ in 0..3 + b.bits(3) {
                    lengths.push(0);
                }
            }
            _ => {
                for _ in 0..11 + b.bits(7) {
                    lengths.push(0);
                }
            }
        }
    }
    (huffman(&lengths[..hlit]), huffman(&lengths[hlit..hlit + hdist]))
}

pub fn inflate(data: &[u8], path: &str) -> Vec<u8> {
    if data.len() < 6 || data[0] & 0x0F != 8 {
        die(&format!("{} has an unsupported zlib stream", path));
    }
    let mut b = Bits { data, pos: 2, bit: 0, path };
    let mut out = Vec::new();
    loop {
        let last = b.bits(1);
        match b.bits(2) {
            0 => {
                b.align();
                if b.pos + 4 > data.len() {
                    die(&format!("{} is truncated", path));
                }
                let len = u16::from_le_bytes([data[b.pos], data[b.pos + 1]]) as usize;
                b.pos += 4;
                if b.pos + len > data.len() {
                    die(&format!("{} is truncated", path));
                }
                out.extend(&data[b.pos..b.pos + len]);
                b.pos += len;
            }
            1 => {
                let (lit, dist) = fixed_tables();
                inflate_block(&mut b, &lit, &dist, &mut out);
            }
            2 => {
                let (lit, dist) = dynamic_tables(&mut b);
                inflate_block(&mut b, &lit, &dist, &mut out);
            }
            _ => die(&format!("{} has an invalid deflate block", path)),
        }
        if last == 1 {
            break;
        }
    }
    out
}

fn deflate_stored(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    let mut chunks = data.chunks(65535).peekable();
    if data.is_empty() {
        out.extend([1, 0, 0, 0xFF, 0xFF]);
    }
    while let Some(chunk) = chunks.next() {
        out.push(if chunks.peek().is_none() { 1 } else { 0 });
        out.extend((chunk.len() as u16).to_le_bytes());
        out.extend((!(chunk.len() as u16)).to_le_bytes());
        out.extend(chunk);
    }
    out.extend(adler32(data).to_be_bytes());
    out
}

struct Ihdr {
    width: usize,
    height: usize,
    bit_depth: u8,
    color_type: u8,
}

fn parse_ihdr(data: &[u8], path: &str) -> Ihdr {
    if data.len() < 33 || data[0..8] != [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A] {
        die(&format!("{} is not a .png file", path));
    }
    if &data[12..16] != b"IHDR" {
        die(&format!("{} is missing its IHDR chunk", path));
    }
    let width = u32::from_be_bytes(data[16..20].try_into().unwrap()) as usize;
    let height = u32::from_be_bytes(data[20..24].try_into().unwrap()) as usize;
    let bit_depth = data[24];
    let color_type = data[25];
    if data[28] != 0 {
        die(&format!("{} is interlaced; re-save without interlacing", path));
    }
    if width == 0 || height == 0 {
        die(&format!("{} has zero size", path));
    }
    match (color_type, bit_depth) {
        (0 | 2 | 4 | 6, 8 | 16) | (3, 8) => {}
        _ => die(&format!("{} has unsupported bit depth {} for color type {}", path, bit_depth, color_type)),
    }
    Ihdr { width, height, bit_depth, color_type }
}

fn raw_channels(color_type: u8) -> usize {
    match color_type {
        0 | 3 => 1,
        4 => 2,
        2 => 3,
        _ => 4,
    }
}

pub fn out_channels(color_type: u8) -> usize {
    if color_type == 3 { 3 } else { raw_channels(color_type) }
}

pub fn png_meta(path: &str) -> (usize, usize, usize) {
    let mut f = fs::File::open(path)
        .unwrap_or_else(|e| die(&format!("cannot open {}: {}", path, e)));
    let mut head = [0u8; 33];
    f.read_exact(&mut head)
        .unwrap_or_else(|e| die(&format!("cannot read {}: {}", path, e)));
    let ihdr = parse_ihdr(&head, path);
    (ihdr.height, ihdr.width, out_channels(ihdr.color_type))
}

fn paeth(a: i32, b: i32, c: i32) -> i32 {
    let p = a + b - c;
    let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
    if pa <= pb && pa <= pc { a } else if pb <= pc { b } else { c }
}

pub fn decode_png(path: &str) -> (usize, usize, usize, Vec<f32>) {
    let data = fs::read(path)
        .unwrap_or_else(|e| die(&format!("cannot read {}: {}", path, e)));
    let ihdr = parse_ihdr(&data, path);
    let mut idat: Vec<u8> = Vec::new();
    let mut palette: Vec<u8> = Vec::new();
    let mut pos = 8;
    while pos + 8 <= data.len() {
        let len = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        if pos + 12 + len > data.len() {
            die(&format!("{} is truncated", path));
        }
        let kind = &data[pos + 4..pos + 8];
        let body = &data[pos + 8..pos + 8 + len];
        let crc = u32::from_be_bytes(data[pos + 8 + len..pos + 12 + len].try_into().unwrap());
        if crc32(&data[pos + 4..pos + 8 + len]) != crc {
            die(&format!("{} has a corrupt {} chunk", path, String::from_utf8_lossy(kind)));
        }
        match kind {
            b"IDAT" => idat.extend(body),
            b"PLTE" => palette = body.to_vec(),
            b"IEND" => break,
            _ => {}
        }
        pos += 12 + len;
    }
    if idat.is_empty() {
        die(&format!("{} has no image data", path));
    }
    let raw = inflate(&idat, path);
    let channels = raw_channels(ihdr.color_type);
    let sample_bytes = if ihdr.bit_depth == 16 { 2 } else { 1 };
    let bpp = channels * sample_bytes;
    let stride = ihdr.width * bpp;
    if raw.len() < ihdr.height * (stride + 1) {
        die(&format!("{} has truncated image data", path));
    }
    let mut lines: Vec<u8> = vec![0; ihdr.height * stride];
    for y in 0..ihdr.height {
        let filter = raw[y * (stride + 1)];
        let row_in = &raw[y * (stride + 1) + 1..y * (stride + 1) + 1 + stride];
        for x in 0..stride {
            let left = if x >= bpp { lines[y * stride + x - bpp] as i32 } else { 0 };
            let up = if y > 0 { lines[(y - 1) * stride + x] as i32 } else { 0 };
            let corner = if x >= bpp && y > 0 { lines[(y - 1) * stride + x - bpp] as i32 } else { 0 };
            let recon = match filter {
                0 => row_in[x] as i32,
                1 => row_in[x] as i32 + left,
                2 => row_in[x] as i32 + up,
                3 => row_in[x] as i32 + (left + up) / 2,
                4 => row_in[x] as i32 + paeth(left, up, corner),
                _ => die(&format!("{} has an invalid scanline filter", path)),
            };
            lines[y * stride + x] = recon as u8;
        }
    }
    let out_c = out_channels(ihdr.color_type);
    let mut vals: Vec<f32> = Vec::with_capacity(ihdr.height * ihdr.width * out_c);
    for y in 0..ihdr.height {
        for x in 0..ihdr.width {
            let base = y * stride + x * bpp;
            if ihdr.color_type == 3 {
                let idx = lines[base] as usize * 3;
                if idx + 2 >= palette.len() {
                    die(&format!("{} has a palette index out of range", path));
                }
                for k in 0..3 {
                    vals.push(palette[idx + k] as f32 / 255.0);
                }
            } else {
                for k in 0..channels {
                    let v = if sample_bytes == 2 {
                        u16::from_be_bytes([lines[base + 2 * k], lines[base + 2 * k + 1]]) as f32 / 65535.0
                    } else {
                        lines[base + k] as f32 / 255.0
                    };
                    vals.push(v);
                }
            }
        }
    }
    (ihdr.height, ihdr.width, out_c, vals)
}

pub fn png_host_buffer(path: &str, shape: &[usize]) -> HostBuffer {
    let (h, w, c, vals) = decode_png(path);
    let expected: Vec<usize> = if c == 1 { vec![h, w] } else { vec![h, w, c] };
    if shape != expected {
        die(&format!("{} changed since compilation: {:?} vs {:?}", path, expected, shape));
    }
    let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    HostBuffer::from_data(vals, Some(dims), None)
}

pub fn encode_png(width: usize, height: usize, channels: usize, pixels: &[u8]) -> Vec<u8> {
    let color_type: u8 = match channels {
        1 => 0,
        3 => 2,
        _ => 6,
    };
    let stride = width * channels;
    let mut raw = Vec::with_capacity(height * (stride + 1));
    for y in 0..height {
        raw.push(0);
        raw.extend(&pixels[y * stride..(y + 1) * stride]);
    }
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::new();
    ihdr.extend((width as u32).to_be_bytes());
    ihdr.extend((height as u32).to_be_bytes());
    ihdr.extend([8, color_type, 0, 0, 0]);
    push_chunk(&mut out, b"IHDR", &ihdr);
    push_chunk(&mut out, b"IDAT", &deflate_stored(&raw));
    push_chunk(&mut out, b"IEND", &[]);
    out
}

fn push_chunk(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
    out.extend((body.len() as u32).to_be_bytes());
    out.extend(kind);
    out.extend(body);
    let mut check = kind.to_vec();
    check.extend(body);
    out.extend(crc32(&check).to_be_bytes());
}

pub fn image_bytes(t: &Tensor) -> (usize, usize, usize, Vec<u8>) {
    let shape = t.shape();
    let (h, w, c) = match shape.len() {
        2 => (shape[0], shape[1], 1),
        _ => (shape[0], shape[1], shape[2]),
    };
    let pixels: Vec<u8> = t.f64_vec().iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    (h, w, c, pixels)
}

pub fn write_png(path: &str, t: &Tensor) {
    let (h, w, c, pixels) = image_bytes(t);
    fs::write(path, encode_png(w, h, c, &pixels))
        .unwrap_or_else(|e| die(&format!("cannot write {}: {}", path, e)));
}

pub fn image_shape(shape: &[usize], what: &str) -> (usize, usize, usize) {
    match shape {
        [h, w] => (*h, *w, 1),
        [h, w, c] if matches!(c, 1 | 3 | 4) => (*h, *w, *c),
        _ => die(&format!("{} expects an image shaped [h, w] or [h, w, c] with 1, 3 or 4 channels, got {:?}", what, shape)),
    }
}

impl Tracer {
    fn interp_matrix(&mut self, out_n: usize, in_n: usize, dtype: Dtype) -> crate::graph::Val {
        let mut vals = vec![0.0f64; out_n * in_n];
        for i in 0..out_n {
            let src = ((i as f64 + 0.5) * in_n as f64 / out_n as f64 - 0.5).max(0.0);
            let i0 = (src.floor() as usize).min(in_n - 1);
            let i1 = (i0 + 1).min(in_n - 1);
            let f = src - i0 as f64;
            vals[i * in_n + i0] += 1.0 - f;
            vals[i * in_n + i1] += f;
        }
        self.emit(OpKind::DenseConst(vals), vec![], vec![out_n, in_n], dtype)
    }

    pub fn resize_image(&mut self, v: BVal, h: usize, w: usize) -> TVal {
        if v.bdims != 0 {
            die("resize inside vmap isn't supported yet");
        }
        if h == 0 || w == 0 {
            die("resize dimensions must be positive");
        }
        let shape = v.val.shape.clone();
        let (ih, iw, c) = image_shape(&shape, "resize");
        let dtype = v.val.dtype;
        let rw = self.interp_matrix(h, ih, dtype);
        let cw = self.interp_matrix(w, iw, dtype);
        let x0 = self.reshape(&v.val, vec![ih, iw * c]);
        let a = self.dot(&rw, &x0, vec![], vec![], vec![1], vec![0]);
        let a2 = self.reshape(&a, vec![h, iw, c]);
        let a3 = self.emit(OpKind::Transpose(vec![1, 0, 2]), vec![a2.id], vec![iw, h, c], dtype);
        let a4 = self.reshape(&a3, vec![iw, h * c]);
        let b = self.dot(&cw, &a4, vec![], vec![], vec![1], vec![0]);
        let b2 = self.reshape(&b, vec![w, h, c]);
        let b3 = self.emit(OpKind::Transpose(vec![1, 0, 2]), vec![b2.id], vec![h, w, c], dtype);
        let out = if shape.len() == 3 { b3 } else { self.reshape(&b3, vec![h, w]) };
        TVal::Tensor(BVal { val: out, bdims: 0 })
    }

    pub fn crop_image(&mut self, v: BVal, top: usize, left: usize, h: usize, w: usize) -> TVal {
        if v.bdims != 0 {
            die("crop inside vmap isn't supported yet");
        }
        if h == 0 || w == 0 {
            die("crop size must be positive");
        }
        let shape = v.val.shape.clone();
        let (ih, iw, _) = image_shape(&shape, "crop");
        if top + h > ih || left + w > iw {
            die(&format!("crop [{}:{}, {}:{}] is out of bounds for shape {:?}", top, top + h, left, left + w, shape));
        }
        let mut s1_shape = shape.clone();
        s1_shape[0] = h;
        let s1 = self.emit(OpKind::Slice(0, top, top + h), vec![v.val.id], s1_shape.clone(), v.val.dtype);
        let mut s2_shape = s1_shape;
        s2_shape[1] = w;
        let s2 = self.emit(OpKind::Slice(1, left, left + w), vec![s1.id], s2_shape, v.val.dtype);
        TVal::Tensor(BVal { val: s2, bdims: 0 })
    }

    pub fn load_png(&mut self, path: &str) -> TVal {
        if let Some(&(_, id)) = self.inputs.iter()
            .find(|(src, _)| matches!(src, InputSource::Image(p) if *p == path)) {
            return TVal::Tensor(BVal { val: self.val(id), bdims: 0 });
        }
        let (h, w, c) = png_meta(path);
        let shape = if c == 1 { vec![h, w] } else { vec![h, w, c] };
        let val = self.emit(OpKind::Input, vec![], shape, Dtype::F32);
        self.inputs.push((InputSource::Image(path.to_string()), val.id));
        TVal::Tensor(BVal { val, bdims: 0 })
    }
}
