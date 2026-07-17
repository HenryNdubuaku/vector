use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use pjrt::HostBuffer;

use crate::die;
use crate::safetensors::{parse_json, Json};

pub fn txt_len(path: &str) -> usize {
    let meta = fs::metadata(path)
        .unwrap_or_else(|e| die(&format!("cannot open {}: {}", path, e)));
    if meta.len() == 0 {
        die(&format!("{} is empty", path));
    }
    meta.len() as usize
}

pub fn txt_host_buffer(path: &str, shape: &[usize]) -> HostBuffer {
    let bytes = fs::read(path)
        .unwrap_or_else(|e| die(&format!("cannot read {}: {}", path, e)));
    if bytes.len() != shape[0] {
        die(&format!("{} changed since compilation: {} bytes vs {}", path, bytes.len(), shape[0]));
    }
    let vals: Vec<f32> = bytes.iter().map(|&b| b as f32).collect();
    HostBuffer::from_data(vals, Some(vec![bytes.len() as i64]), None)
}

pub fn write_txt(path: &str, vals: &[f64]) {
    let bytes: Vec<u8> = vals.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();
    fs::write(path, bytes)
        .unwrap_or_else(|e| die(&format!("cannot write {}: {}", path, e)));
}

pub fn bytes_to_string(vals: &[f64]) -> String {
    let bytes: Vec<u8> = vals.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

struct Bpe {
    id_to_token: HashMap<u32, String>,
    token_to_id: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    add_prefix_space: bool,
    words: Mutex<HashMap<String, Arc<Vec<u32>>>>,
}

fn byte_unicode() -> ([char; 256], HashMap<char, u8>) {
    let mut byte_to_char = ['\0'; 256];
    let mut mapped = [false; 256];
    for b in (0x21..=0x7E).chain(0xA1..=0xAC).chain(0xAE..=0xFF) {
        byte_to_char[b as usize] = char::from_u32(b).unwrap();
        mapped[b as usize] = true;
    }
    let mut n = 0;
    for b in 0..256 {
        if !mapped[b] {
            byte_to_char[b] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    let mut char_to_byte = HashMap::new();
    for (b, &c) in byte_to_char.iter().enumerate() {
        char_to_byte.insert(c, b as u8);
    }
    (byte_to_char, char_to_byte)
}

fn field<'a>(fields: &'a [(String, Json)], name: &str) -> Option<&'a Json> {
    fields.iter().find(|(k, _)| k == name).map(|(_, v)| v)
}

fn check_byte_level(pre: Option<&Json>, path: &str) -> bool {
    let Some(Json::Obj(fields)) = pre else {
        die(&format!("{} has no pre_tokenizer; only byte-level bpe tokenizers are supported yet", path));
    };
    let Some(Json::Str(kind)) = field(fields, "type") else {
        die(&format!("{} has a malformed pre_tokenizer", path));
    };
    if kind != "ByteLevel" {
        die(&format!("{} uses a {} pre_tokenizer; only byte-level bpe tokenizers (gpt-2 family) are supported yet", path, kind));
    }
    matches!(field(fields, "add_prefix_space"), Some(Json::Bool(true)))
}

fn load_bpe(path: &str) -> Bpe {
    let bytes = fs::read(path)
        .unwrap_or_else(|e| die(&format!("cannot read {}: {}", path, e)));
    let Json::Obj(root) = parse_json(&bytes, path) else {
        die(&format!("{} is not a json object", path));
    };
    let Some(Json::Obj(model)) = field(&root, "model") else {
        die(&format!("{} has no model section", path));
    };
    if let Some(Json::Str(kind)) = field(model, "type") {
        if kind != "BPE" {
            die(&format!("{} is a {} tokenizer; only bpe is supported yet", path, kind));
        }
    }
    let add_prefix_space = check_byte_level(field(&root, "pre_tokenizer"), path);
    let Some(Json::Obj(vocab)) = field(model, "vocab") else {
        die(&format!("{} has no vocab", path));
    };
    let mut token_to_id = HashMap::new();
    let mut id_to_token = HashMap::new();
    for (token, id) in vocab {
        let Json::Num(id) = id else {
            die(&format!("{} has a malformed vocab", path));
        };
        token_to_id.insert(token.clone(), *id as u32);
        id_to_token.insert(*id as u32, token.clone());
    }
    if let Some(Json::Arr(added)) = field(&root, "added_tokens") {
        for entry in added {
            let Json::Obj(fields) = entry else { continue };
            if let (Some(Json::Num(id)), Some(Json::Str(content))) = (field(fields, "id"), field(fields, "content")) {
                token_to_id.insert(content.clone(), *id as u32);
                id_to_token.insert(*id as u32, content.clone());
            }
        }
    }
    let Some(Json::Arr(merges)) = field(model, "merges") else {
        die(&format!("{} has no merges", path));
    };
    let mut ranks = HashMap::new();
    for (rank, merge) in merges.iter().enumerate() {
        let (a, b) = match merge {
            Json::Str(s) => match s.split_once(' ') {
                Some((a, b)) => (a.to_string(), b.to_string()),
                None => die(&format!("{} has a malformed merge: {}", path, s)),
            },
            Json::Arr(pair) => match (pair.first(), pair.get(1)) {
                (Some(Json::Str(a)), Some(Json::Str(b))) => (a.clone(), b.clone()),
                _ => die(&format!("{} has a malformed merge entry", path)),
            },
            _ => die(&format!("{} has a malformed merges list", path)),
        };
        ranks.insert((a, b), rank);
    }
    let (byte_to_char, char_to_byte) = byte_unicode();
    Bpe {
        id_to_token,
        token_to_id,
        ranks,
        byte_to_char,
        char_to_byte,
        add_prefix_space,
        words: Mutex::new(HashMap::new()),
    }
}

fn is_other(c: char) -> bool {
    !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric()
}

fn split_words(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut words = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let mut end = i + 1;
        if chars[i] == '\'' && i + 1 < chars.len() {
            let two: String = chars[i + 1..].iter().take(2).collect();
            if two.starts_with('s') || two.starts_with('t') || two.starts_with('m') || two.starts_with('d') {
                end = i + 2;
            }
            if two == "re" || two == "ve" || two == "ll" {
                end = i + 3;
            }
        }
        if end == i + 1 {
            let start = if chars[i] == ' ' && i + 1 < chars.len() { i + 1 } else { i };
            let class: Option<fn(char) -> bool> = if chars[start].is_alphabetic() {
                Some(char::is_alphabetic)
            } else if chars[start].is_numeric() {
                Some(char::is_numeric)
            } else if is_other(chars[start]) {
                Some(is_other)
            } else {
                None
            };
            if let Some(class) = class {
                end = start + 1;
                while end < chars.len() && class(chars[end]) {
                    end += 1;
                }
            } else {
                end = i + 1;
                while end < chars.len() && chars[end].is_whitespace() {
                    end += 1;
                }
                if end < chars.len() && end - i >= 2 {
                    end -= 1;
                }
            }
        }
        words.push(chars[i..end].iter().collect());
        i = end;
    }
    words
}

impl Bpe {
    fn encode_word(&self, word: &str, path: &str) -> Arc<Vec<u32>> {
        if let Some(ids) = self.words.lock().unwrap().get(word) {
            return ids.clone();
        }
        let mut parts: Vec<String> = word
            .bytes()
            .map(|b| self.byte_to_char[b as usize].to_string())
            .collect();
        loop {
            let mut best: Option<(usize, usize)> = None;
            for p in 0..parts.len() - 1 {
                if let Some(&r) = self.ranks.get(&(parts[p].clone(), parts[p + 1].clone())) {
                    if best.is_none_or(|(br, _)| r < br) {
                        best = Some((r, p));
                    }
                }
            }
            let Some((_, p)) = best else { break };
            let merged = format!("{}{}", parts[p], parts[p + 1]);
            parts.splice(p..p + 2, [merged]);
        }
        let ids: Vec<u32> = parts
            .iter()
            .map(|t| {
                *self.token_to_id.get(t).unwrap_or_else(|| {
                    die(&format!("{} has no token for {:?}; the vocab doesn't cover all bytes", path, t))
                })
            })
            .collect();
        let ids = Arc::new(ids);
        self.words.lock().unwrap().insert(word.to_string(), ids.clone());
        ids
    }

    fn encode(&self, text: &str, path: &str) -> Vec<u32> {
        let text = if self.add_prefix_space && !text.starts_with(' ') {
            format!(" {}", text)
        } else {
            text.to_string()
        };
        let mut ids = Vec::new();
        for word in split_words(&text) {
            ids.extend(self.encode_word(&word, path).iter());
        }
        ids
    }

    fn decode(&self, ids: &[u32], path: &str) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(token) = self.id_to_token.get(&id) else {
                die(&format!("id {} is not in {}", id, path));
            };
            for c in token.chars() {
                match self.char_to_byte.get(&c) {
                    Some(&b) => bytes.push(b),
                    None => bytes.extend(c.to_string().as_bytes()),
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

static TOKENIZERS: Mutex<Vec<(String, SystemTime, Arc<Bpe>)>> = Mutex::new(Vec::new());
static ENCODED: Mutex<Vec<(String, String, Arc<Vec<u32>>)>> = Mutex::new(Vec::new());

fn tokenizer(path: &str) -> Arc<Bpe> {
    let mtime = fs::metadata(path)
        .and_then(|m| m.modified())
        .unwrap_or_else(|e| die(&format!("cannot open {}: {}", path, e)));
    let mut cache = TOKENIZERS.lock().unwrap();
    if let Some((_, t, bpe)) = cache.iter().find(|(p, _, _)| p == path) {
        if *t == mtime {
            return bpe.clone();
        }
    }
    let bpe = Arc::new(load_bpe(path));
    cache.retain(|(p, _, _)| p != path);
    cache.push((path.to_string(), mtime, bpe.clone()));
    bpe
}

pub fn check_tokenizer(path: &str) {
    tokenizer(path);
}

pub fn encode_file(txt: &str, tok: &str) -> Arc<Vec<u32>> {
    let mut cache = ENCODED.lock().unwrap();
    if let Some((_, _, ids)) = cache.iter().find(|(t, k, _)| t == txt && k == tok) {
        return ids.clone();
    }
    let bytes = fs::read(txt)
        .unwrap_or_else(|e| die(&format!("cannot read {}: {}", txt, e)));
    let content = String::from_utf8_lossy(&bytes);
    let ids = Arc::new(tokenizer(tok).encode(&content, tok));
    if ids.is_empty() {
        die(&format!("{} tokenized to nothing", txt));
    }
    cache.push((txt.to_string(), tok.to_string(), ids.clone()));
    ids
}

pub fn decode_ids(vals: &[f64], tok: &str) -> String {
    let ids: Vec<u32> = vals.iter().map(|&v| v.round().max(0.0) as u32).collect();
    tokenizer(tok).decode(&ids, tok)
}

pub fn tokens_host_buffer(txt: &str, tok: &str, shape: &[usize]) -> HostBuffer {
    let ids = encode_file(txt, tok);
    if ids.len() != shape[0] {
        die(&format!("{} changed since compilation: {} tokens vs {}", txt, ids.len(), shape[0]));
    }
    let vals: Vec<f32> = ids.iter().map(|&t| t as f32).collect();
    HostBuffer::from_data(vals, Some(vec![ids.len() as i64]), None)
}
