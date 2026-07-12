use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::panic::{catch_unwind, AssertUnwindSafe};

use pjrt::{HostBuffer, LoadedExecutable};

use crate::die;
use crate::graph::Dtype;
use crate::runtime::{Engine, Tensor};
use crate::safetensors::{parse_json, Json};
use crate::VectorError;

struct Port {
    shape: Vec<usize>,
    dtype: Dtype,
}

fn parse_tensor_type(s: &str, path: &str) -> Port {
    let inner = s.trim()
        .strip_prefix("tensor<")
        .and_then(|t| t.strip_suffix('>'))
        .unwrap_or_else(|| die(&format!("{} has an unrecognized type {}", path, s)));
    let mut parts: Vec<&str> = inner.split('x').collect();
    let dtype = match parts.pop() {
        Some("f32") => Dtype::F32,
        Some("f64") => Dtype::F64,
        t => die(&format!("{} has an unservable dtype {:?}", path, t)),
    };
    let shape = parts.iter()
        .map(|d| d.parse().unwrap_or_else(|_| die(&format!("{} has an unrecognized type {}", path, s))))
        .collect();
    Port { shape, dtype }
}

fn parse_signature(mlir: &str, path: &str) -> (Vec<Port>, Vec<Port>) {
    let line = mlir.lines()
        .find(|l| l.contains("func.func @main"))
        .unwrap_or_else(|| die(&format!("{} has no @main function", path)));
    let open = line.find('(').unwrap_or_else(|| die(&format!("{} has a malformed signature", path)));
    let close = line.find(')').unwrap_or_else(|| die(&format!("{} has a malformed signature", path)));
    let params: Vec<Port> = line[open + 1..close]
        .split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| {
            let ty = p.split(':').nth(1)
                .unwrap_or_else(|| die(&format!("{} has a malformed signature", path)));
            parse_tensor_type(ty, path)
        })
        .collect();
    let outputs = match line[close..].find("-> (") {
        Some(arrow) => {
            let rest = &line[close + arrow + 4..];
            let end = rest.find(')')
                .unwrap_or_else(|| die(&format!("{} has a malformed signature", path)));
            rest[..end].split(", ").map(|t| parse_tensor_type(t, path)).collect()
        }
        None => Vec::new(),
    };
    (params, outputs)
}

fn port_text(p: &Port) -> String {
    let dims: Vec<String> = p.shape.iter().map(|d| d.to_string()).collect();
    let mut s = dims.join("x");
    if !s.is_empty() {
        s.push('x');
    }
    s.push_str(p.dtype.name());
    s
}

fn fill(v: &Json, shape: &[usize], out: &mut Vec<f64>) {
    if shape.is_empty() {
        match v {
            Json::Num(n) => out.push(*n),
            _ => die("request inputs must be nested arrays of numbers"),
        }
        return;
    }
    let Json::Arr(items) = v else {
        die("request inputs must be nested arrays of numbers");
    };
    if items.len() != shape[0] {
        die(&format!("request input has {} elements where {} were expected", items.len(), shape[0]));
    }
    for item in items {
        fill(item, &shape[1..], out);
    }
}

fn feed(v: &Json, port: &Port) -> HostBuffer {
    let mut vals = Vec::new();
    fill(v, &port.shape, &mut vals);
    let dims: Vec<i64> = port.shape.iter().map(|&d| d as i64).collect();
    match port.dtype {
        Dtype::F64 => HostBuffer::from_data(vals, Some(dims), None),
        _ => {
            let vals: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
            HostBuffer::from_data(vals, Some(dims), None)
        }
    }
}

fn number_json(v: f64, f32ish: bool) -> String {
    if !v.is_finite() {
        return "null".to_string();
    }
    if f32ish {
        format!("{}", v as f32)
    } else {
        format!("{}", v)
    }
}

fn tensor_json(t: &Tensor) -> String {
    fn rec(vals: &[f64], shape: &[usize], f32ish: bool) -> String {
        if shape.is_empty() {
            return number_json(vals[0], f32ish);
        }
        let inner: usize = shape[1..].iter().product::<usize>().max(1);
        let parts: Vec<String> = (0..shape[0])
            .map(|i| rec(&vals[i * inner..(i + 1) * inner], &shape[1..], f32ish))
            .collect();
        format!("[{}]", parts.join(","))
    }
    rec(&t.f64_vec(), t.shape(), t.graph_dtype() == Dtype::F32)
}

fn infer(body: &[u8], inputs: &[Port], engine: &Engine, executable: &LoadedExecutable) -> String {
    let Json::Obj(fields) = parse_json(body, "request") else {
        die("request body must be a json object like {\"inputs\": [...]}");
    };
    let Some((_, Json::Arr(given))) = fields.iter().find(|(k, _)| k == "inputs") else {
        die("request body must have an \"inputs\" array");
    };
    if given.len() != inputs.len() {
        die(&format!("model takes {} inputs, request has {}", inputs.len(), given.len()));
    }
    let feeds: Vec<HostBuffer> = given.iter().zip(inputs).map(|(v, p)| feed(v, p)).collect();
    let results = engine.run(executable, feeds);
    let outs: Vec<String> = results.iter().map(tensor_json).collect();
    format!("{{\"outputs\":[{}]}}", outs.join(","))
}

fn respond(stream: &mut TcpStream, status: &str, body: &str) {
    let _ = write!(
        stream,
        "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status, body.len(), body
    );
}

fn handle(stream: &mut TcpStream, signature: &str, inputs: &[Port], engine: &Engine, executable: &LoadedExecutable) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    let method = request_line.split_whitespace().next().unwrap_or("").to_string();
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
            break;
        }
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    if method == "GET" {
        respond(stream, "200 OK", signature);
        return;
    }
    if method != "POST" {
        respond(stream, "405 Method Not Allowed", "{\"error\":\"use GET / for the signature or POST / for inference\"}");
        return;
    }
    let mut body = vec![0u8; content_length];
    if reader.read_exact(&mut body).is_err() {
        respond(stream, "400 Bad Request", "{\"error\":\"truncated request body\"}");
        return;
    }
    let outcome = catch_unwind(AssertUnwindSafe(|| infer(&body, inputs, engine, executable)));
    match outcome {
        Ok(response) => respond(stream, "200 OK", &response),
        Err(e) => {
            let msg = e.downcast_ref::<VectorError>()
                .map(|v| v.0.clone())
                .unwrap_or_else(|| "internal error".to_string());
            let escaped = msg.replace('\\', "\\\\").replace('"', "\\\"");
            respond(stream, "400 Bad Request", &format!("{{\"error\":\"{}\"}}", escaped));
        }
    }
}

pub fn serve(path: &str, port: u16) {
    let mlir = fs::read_to_string(path)
        .unwrap_or_else(|e| die(&format!("cannot read {}: {}", path, e)));
    let (inputs, outputs) = parse_signature(&mlir, path);
    let engine = Engine::new();
    let executable = engine.prepare(&mlir);
    let in_texts: Vec<String> = inputs.iter().map(|p| format!("\"{}\"", port_text(p))).collect();
    let out_texts: Vec<String> = outputs.iter().map(|p| format!("\"{}\"", port_text(p))).collect();
    let signature = format!("{{\"inputs\":[{}],\"outputs\":[{}]}}", in_texts.join(","), out_texts.join(","));
    let listener = TcpListener::bind(("0.0.0.0", port))
        .unwrap_or_else(|e| die(&format!("cannot listen on port {}: {}", port, e)));
    println!("serving {} on http://0.0.0.0:{}", path, port);
    println!("  signature: {}", signature);
    println!("  infer with: curl -d '{{\"inputs\": [...]}}' http://127.0.0.1:{}", port);
    for stream in listener.incoming() {
        if let Ok(mut stream) = stream {
            handle(&mut stream, &signature, &inputs, &engine, &executable);
        }
    }
}
