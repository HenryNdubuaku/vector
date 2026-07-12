use std::fs;
use std::io::{Read, Seek, SeekFrom};

use pjrt::HostBuffer;

use crate::die;
use crate::graph::{BVal, Dtype, InputSource, ModTag, OpKind, TVal, Val};
use crate::runtime::Tensor;
use crate::trace::Tracer;

pub struct StTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub begin: usize,
    pub end: usize,
}

pub struct StMeta {
    pub tensors: Vec<StTensor>,
    pub metadata: Vec<(String, String)>,
    pub data_start: usize,
}

enum Json {
    Str(String),
    Num(f64),
    Other,
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
    path: &'a str,
}

impl<'a> JsonParser<'a> {
    fn fail(&self, what: &str) -> ! {
        die(&format!("malformed safetensors header in {}: {}", self.path, what))
    }

    fn peek(&self) -> u8 {
        match self.bytes.get(self.pos) {
            Some(&b) => b,
            None => self.fail("unexpected end"),
        }
    }

    fn skip_ws(&mut self) {
        while matches!(self.bytes.get(self.pos), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, b: u8) {
        self.skip_ws();
        if self.peek() != b {
            self.fail(&format!("expected '{}'", b as char));
        }
        self.pos += 1;
    }

    fn value(&mut self) -> Json {
        self.skip_ws();
        match self.peek() {
            b'"' => Json::Str(self.string()),
            b'{' => self.object(),
            b'[' => self.array(),
            b't' => { self.literal("true"); Json::Other }
            b'f' => { self.literal("false"); Json::Other }
            b'n' => { self.literal("null"); Json::Other }
            _ => self.number(),
        }
    }

    fn literal(&mut self, word: &str) {
        if self.bytes[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
        } else {
            self.fail("unknown literal");
        }
    }

    fn number(&mut self) -> Json {
        let start = self.pos;
        while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')) {
            self.pos += 1;
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos]).unwrap_or_else(|_| self.fail("bad number"));
        match s.parse() {
            Ok(n) => Json::Num(n),
            Err(_) => self.fail("bad number"),
        }
    }

    fn hex4(&mut self) -> u32 {
        let mut v = 0u32;
        for _ in 0..4 {
            let d = (self.peek() as char).to_digit(16).unwrap_or_else(|| self.fail("bad \\u escape"));
            v = v * 16 + d;
            self.pos += 1;
        }
        v
    }

    fn string(&mut self) -> String {
        self.expect(b'"');
        let mut out: Vec<u8> = Vec::new();
        loop {
            let b = self.peek();
            self.pos += 1;
            match b {
                b'"' => break,
                b'\\' => {
                    let e = self.peek();
                    self.pos += 1;
                    match e {
                        b'"' => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'/' => out.push(b'/'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'u' => {
                            let mut cp = self.hex4();
                            if (0xD800..=0xDBFF).contains(&cp) {
                                self.expect(b'\\');
                                self.expect(b'u');
                                let low = self.hex4();
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                            }
                            let ch = char::from_u32(cp).unwrap_or_else(|| self.fail("bad \\u escape"));
                            let mut buf = [0u8; 4];
                            out.extend(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        _ => self.fail("unknown escape"),
                    }
                }
                _ => out.push(b),
            }
        }
        String::from_utf8(out).unwrap_or_else(|_| self.fail("invalid utf-8"))
    }

    fn array(&mut self) -> Json {
        self.expect(b'[');
        self.skip_ws();
        let mut items = Vec::new();
        if self.peek() == b']' {
            self.pos += 1;
            return Json::Arr(items);
        }
        loop {
            items.push(self.value());
            self.skip_ws();
            match self.peek() {
                b',' => { self.pos += 1; }
                b']' => { self.pos += 1; break; }
                _ => self.fail("expected ',' or ']'"),
            }
        }
        Json::Arr(items)
    }

    fn object(&mut self) -> Json {
        self.expect(b'{');
        self.skip_ws();
        let mut fields = Vec::new();
        if self.peek() == b'}' {
            self.pos += 1;
            return Json::Obj(fields);
        }
        loop {
            self.skip_ws();
            let key = self.string();
            self.expect(b':');
            fields.push((key, self.value()));
            self.skip_ws();
            match self.peek() {
                b',' => { self.pos += 1; }
                b'}' => { self.pos += 1; break; }
                _ => self.fail("expected ',' or '}'"),
            }
        }
        Json::Obj(fields)
    }
}

fn json_usize(v: &Json, path: &str, what: &str) -> usize {
    match v {
        Json::Num(n) if n.fract() == 0.0 && *n >= 0.0 => *n as usize,
        _ => die(&format!("malformed safetensors header in {}: bad {}", path, what)),
    }
}

pub fn read_meta(path: &str) -> StMeta {
    let mut f = fs::File::open(path)
        .unwrap_or_else(|e| die(&format!("cannot open {}: {}", path, e)));
    let file_len = f.metadata()
        .unwrap_or_else(|e| die(&format!("cannot read {}: {}", path, e)))
        .len() as usize;
    let mut intro = [0u8; 8];
    f.read_exact(&mut intro)
        .unwrap_or_else(|e| die(&format!("cannot read {}: {}", path, e)));
    let header_len = u64::from_le_bytes(intro) as usize;
    if header_len == 0 || 8 + header_len > file_len {
        die(&format!("{} is not a .safetensors file", path));
    }
    let mut header = vec![0u8; header_len];
    f.read_exact(&mut header)
        .unwrap_or_else(|e| die(&format!("cannot read {}: {}", path, e)));
    let mut p = JsonParser { bytes: &header, pos: 0, path };
    let entries = match p.value() {
        Json::Obj(entries) => entries,
        _ => die(&format!("malformed safetensors header in {}: expected an object", path)),
    };
    let data_start = 8 + header_len;
    let mut tensors = Vec::new();
    let mut metadata = Vec::new();
    for (key, val) in entries {
        if key == "__metadata__" {
            let Json::Obj(kv) = val else {
                die(&format!("malformed safetensors header in {}: bad __metadata__", path));
            };
            for (k, v) in kv {
                if let Json::Str(s) = v {
                    metadata.push((k, s));
                }
            }
            continue;
        }
        let Json::Obj(fields) = val else {
            die(&format!("malformed safetensors header in {}: bad tensor entry '{}'", path, key));
        };
        let mut dtype = None;
        let mut shape = None;
        let mut offsets = None;
        for (k, v) in &fields {
            match (k.as_str(), v) {
                ("dtype", Json::Str(s)) => {
                    dtype = Some(match s.as_str() {
                        "F32" => Dtype::F32,
                        "F64" => Dtype::F64,
                        other => die(&format!("unsupported dtype {} in {} (vector reads F32/F64)", other, path)),
                    });
                }
                ("shape", Json::Arr(items)) => {
                    shape = Some(items.iter().map(|d| json_usize(d, path, "shape")).collect::<Vec<usize>>());
                }
                ("data_offsets", Json::Arr(two)) if two.len() == 2 => {
                    offsets = Some((json_usize(&two[0], path, "data_offsets"), json_usize(&two[1], path, "data_offsets")));
                }
                _ => {}
            }
        }
        let (Some(dtype), Some(shape), Some((begin, end))) = (dtype, shape, offsets) else {
            die(&format!("malformed safetensors header in {}: tensor '{}' is missing fields", path, key));
        };
        let size = if dtype == Dtype::F32 { 4 } else { 8 };
        let count: usize = shape.iter().product();
        if end < begin || data_start + end > file_len || end - begin != count * size {
            die(&format!("tensor '{}' in {} has inconsistent offsets", key, path));
        }
        tensors.push(StTensor { name: key, dtype, shape, begin, end });
    }
    StMeta { tensors, metadata, data_start }
}

pub fn tensor_host_buffer(path: &str, name: &str, shape: &[usize], dtype: Dtype) -> HostBuffer {
    let meta = read_meta(path);
    let t = meta.tensors.iter().find(|t| t.name == name)
        .unwrap_or_else(|| die(&format!("{} changed since compilation: tensor '{}' is missing", path, name)));
    if t.shape != shape || t.dtype != dtype {
        die(&format!("{} changed since compilation: {:?} {} vs {:?} {}",
                     path, t.shape, t.dtype.name(), shape, dtype.name()));
    }
    let mut f = fs::File::open(path)
        .unwrap_or_else(|e| die(&format!("cannot open {}: {}", path, e)));
    f.seek(SeekFrom::Start((meta.data_start + t.begin) as u64))
        .unwrap_or_else(|e| die(&format!("cannot read {}: {}", path, e)));
    let mut data = vec![0u8; t.end - t.begin];
    f.read_exact(&mut data)
        .unwrap_or_else(|e| die(&format!("{} is truncated: {}", path, e)));
    crate::npy::host_buffer(dtype, &t.shape, &data)
}

fn json_escape(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct SaveSpec {
    pub path: String,
    pub names: Vec<String>,
    pub vals: Vec<Val>,
    pub metadata: Vec<(String, String)>,
    pub value: TVal,
}

pub fn write_save(spec: &SaveSpec, tensors: &[Tensor]) {
    if spec.path.ends_with(".npy") {
        crate::npy::write_npy(&spec.path, &tensors[0]);
        return;
    }
    if spec.path.ends_with(".csv") {
        crate::table::write_csv(spec, tensors);
        return;
    }
    let mut parts: Vec<String> = Vec::new();
    if !spec.metadata.is_empty() {
        let kv: Vec<String> = spec.metadata.iter()
            .map(|(k, v)| format!("\"{}\":\"{}\"", json_escape(k), json_escape(v)))
            .collect();
        parts.push(format!("\"__metadata__\":{{{}}}", kv.join(",")));
    }
    let mut data: Vec<u8> = Vec::new();
    for (name, t) in spec.names.iter().zip(tensors) {
        let begin = data.len();
        data.extend(t.le_bytes());
        let dims: Vec<String> = t.shape().iter().map(|d| d.to_string()).collect();
        let dtype = match t.graph_dtype() {
            Dtype::F32 => "F32",
            Dtype::F64 => "F64",
            _ => unreachable!("saves are checked at trace time"),
        };
        parts.push(format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            json_escape(name), dtype, dims.join(","), begin, data.len()
        ));
    }
    let mut header = format!("{{{}}}", parts.join(","));
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.as_bytes());
    bytes.extend(&data);
    fs::write(&spec.path, bytes)
        .unwrap_or_else(|e| die(&format!("cannot write {}: {}", spec.path, e)));
}

fn check_leaf(b: &BVal, file: &str) {
    if b.val.dtype == Dtype::I1 {
        die(&format!("cannot save booleans to {}; use where to select values", file));
    }
    if b.bdims != 0 {
        die("save inside vmap isn't supported");
    }
}

fn collect_save(v: &TVal, path: &str, spec: &mut SaveSpec, file: &str) {
    match v {
        TVal::Tensor(b) => {
            check_leaf(b, file);
            spec.names.push(path.to_string());
            spec.vals.push(b.val.clone());
        }
        TVal::Record(tag, fields) => {
            let key = if path.is_empty() { ".".to_string() } else { path.to_string() };
            let desc = match tag {
                None => "record".to_string(),
                Some(t) => {
                    let statics: String = t.statics.iter().map(|(k, v)| format!(" {}={}", k, v)).collect();
                    format!("module {}{}", t.module, statics)
                }
            };
            spec.metadata.push((key, desc));
            for (k, f) in fields {
                let child = if path.is_empty() { k.clone() } else { format!("{}.{}", path, k) };
                collect_save(f, &child, spec, file);
            }
        }
    }
}

fn field_component(part: &str, full: &str) -> String {
    let mapped = if part.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        format!("_{}", part)
    } else {
        part.to_string()
    };
    if mapped.is_empty() || !mapped.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        die(&format!("tensor name '{}' is not a loadable field path", full));
    }
    mapped
}

fn insert_leaf(node: &mut TVal, comps: &[String], leaf: TVal, full: &str) {
    let TVal::Record(_, fields) = node else {
        die(&format!("conflicting tensor names at '{}'", full));
    };
    let (head, rest) = comps.split_first().unwrap();
    if rest.is_empty() {
        if fields.iter().any(|(k, _)| k == head) {
            die(&format!("conflicting tensor names at '{}'", full));
        }
        fields.push((head.clone(), leaf));
        return;
    }
    if !fields.iter().any(|(k, _)| k == head) {
        fields.push((head.clone(), TVal::Record(None, Vec::new())));
    }
    let child = fields.iter_mut().find(|(k, _)| k == head).map(|(_, v)| v).unwrap();
    insert_leaf(child, rest, leaf, full);
}

fn parse_desc(desc: &str) -> Option<Option<(String, Vec<(String, f64)>)>> {
    if desc == "record" {
        return Some(None);
    }
    let mut it = desc.split(' ');
    if it.next()? != "module" {
        return None;
    }
    let name = it.next()?.to_string();
    let mut statics = Vec::new();
    for kv in it {
        let (k, v) = kv.split_once('=')?;
        statics.push((k.to_string(), v.parse().ok()?));
    }
    Some(Some((name, statics)))
}

fn set_tag(node: &mut TVal, comps: &[String], tag: Option<(String, Vec<(String, f64)>)>) {
    if comps.is_empty() {
        if let TVal::Record(slot, _) = node {
            *slot = tag.map(|(module, statics)| ModTag { module, statics });
        }
        return;
    }
    let TVal::Record(_, fields) = node else { return };
    if !fields.iter().any(|(k, _)| k == &comps[0]) {
        fields.push((comps[0].clone(), TVal::Record(None, Vec::new())));
    }
    let child = fields.iter_mut().find(|(k, _)| k == &comps[0]).map(|(_, v)| v).unwrap();
    set_tag(child, &comps[1..], tag);
}

impl Tracer {
    pub fn plan_save(&mut self, v: &TVal, path: &str) {
        if self.region_depth > 0 {
            die("save inside a for loop isn't supported (loops compile to one XLA while op); save after the loop");
        }
        if self.saves.iter().any(|s| s.path == path) {
            die(&format!("duplicate save to {}", path));
        }
        let mut spec = SaveSpec { path: path.to_string(), names: Vec::new(), vals: Vec::new(), metadata: Vec::new(), value: v.clone() };
        if path.ends_with(".npy") {
            match v {
                TVal::Tensor(b) => {
                    check_leaf(b, path);
                    spec.names.push(String::new());
                    spec.vals.push(b.val.clone());
                }
                TVal::Record(..) => die("save to .npy expects a tensor; records save to .safetensors"),
            }
        } else if path.ends_with(".safetensors") {
            match v {
                TVal::Tensor(_) => die("save to .safetensors expects a record or module instance; tensors save to .npy"),
                TVal::Record(..) => collect_save(v, "", &mut spec, path),
            }
        } else if path.ends_with(".csv") {
            spec = crate::table::csv_save_spec(self, v, path);
        } else {
            die("save expects a path ending in .npy, .safetensors or .csv");
        }
        self.saves.push(spec);
    }

    pub fn load_safetensors(&mut self, path: &str) -> TVal {
        let meta = read_meta(path);
        if meta.tensors.is_empty() {
            die(&format!("{} has no tensors", path));
        }
        let mut root = TVal::Record(None, Vec::new());
        for t in &meta.tensors {
            let val = self.safetensors_input(path, t);
            let comps: Vec<String> = t.name.split('.').map(|c| field_component(c, &t.name)).collect();
            insert_leaf(&mut root, &comps, TVal::Tensor(BVal { val, bdims: 0 }), &t.name);
        }
        for (mpath, desc) in &meta.metadata {
            let Some(tag) = parse_desc(desc) else { continue };
            if let Some((module, _)) = &tag {
                if !self.modules.contains_key(module) {
                    die(&format!("{} was saved from module {}; define module {} before loading", path, module, module));
                }
            }
            let comps: Vec<String> = if mpath == "." {
                Vec::new()
            } else {
                mpath.split('.').map(|c| field_component(c, mpath)).collect()
            };
            set_tag(&mut root, &comps, tag);
        }
        root
    }

    fn safetensors_input(&mut self, path: &str, t: &StTensor) -> Val {
        if let Some(&(_, id)) = self.inputs.iter().find(|(src, _)| {
            matches!(src, InputSource::Safetensors(p, n) if p == path && n == &t.name)
        }) {
            return self.val(id);
        }
        let val = self.emit(OpKind::Input, vec![], t.shape.clone(), t.dtype);
        self.inputs.push((InputSource::Safetensors(path.to_string(), t.name.clone()), val.id));
        val
    }
}
