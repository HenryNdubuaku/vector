mod emit;
mod grad;
mod graph;
mod lexer;
mod linear;
mod npy;
mod parser;
mod runtime;
mod trace;

use std::collections::HashMap;
use std::env;
use std::fs;
use std::process::{exit, Command};

use emit::build_module;
use lexer::lex;
use npy::InputSpec;
use parser::Parser;
use runtime::{execute, format_tensor};
use trace::Tracer;

const USAGE: &str = "usage: vector <command>

  run <file.vec>      compile and execute
  build <file.vec>    print StableHLO to stdout
  setup               download the PJRT CPU plugin to ~/.vector
  version             print version";

fn die(msg: &str) -> ! {
    eprintln!("{}", msg);
    exit(1);
}

fn home() -> String {
    env::var("HOME").unwrap_or_else(|_| die("HOME is not set"))
}

fn plugin_file() -> &'static str {
    if cfg!(target_os = "macos") { "libpjrt_cpu.dylib" } else { "libpjrt_cpu.so" }
}

fn plugin_path() -> String {
    if let Ok(p) = env::var("PJRT_PLUGIN_PATH") {
        return p;
    }
    let path = format!("{}/.vector/{}", home(), plugin_file());
    if fs::metadata(&path).is_err() {
        die(&format!("PJRT plugin not found at {}; run `vector setup` or set PJRT_PLUGIN_PATH", path));
    }
    path
}

fn compile(path: &str) -> (String, Vec<InputSpec>, Vec<Option<String>>) {
    let src = fs::read_to_string(path)
        .unwrap_or_else(|e| die(&format!("cannot read file: {}", e)));
    let lexed = lex(&src);
    let mut p = Parser {
        toks: lexed.toks,
        cols: lexed.cols,
        lines: lexed.lines,
        pos: 0,
        fns: HashMap::new(),
        modules: HashMap::new(),
    };
    let prog = p.program();
    let mut modules = linear::stdlib_modules();
    modules.extend(prog.modules);
    let mut tracer = Tracer {
        nodes: Vec::new(),
        prints: Vec::new(),
        inputs: Vec::new(),
        modules,
        statics: Vec::new(),
        rng: 0x243F6A8885A308D3,
        claimed: std::collections::HashSet::new(),
        region_depth: 0,
        grad_depth: 0,
        interned: vec![HashMap::new()],
    };
    tracer.trace(&prog.main, &HashMap::new(), &prog.fns);
    let outputs: Vec<_> = tracer.prints.iter().map(|(_, v)| v.clone()).collect();
    let labels: Vec<Option<String>> = tracer.prints.iter().map(|(l, _)| l.clone()).collect();
    let specs: Vec<InputSpec> = tracer.inputs.iter()
        .map(|&(ref path, id)| InputSpec {
            path: path.clone(),
            shape: tracer.nodes[id].shape.clone(),
            dtype: tracer.nodes[id].dtype,
        })
        .collect();
    (build_module(&tracer, &outputs), specs, labels)
}

fn run(path: &str) {
    let (module, specs, labels) = compile(path);
    for (label, tensor) in labels.iter().zip(execute(&module, &specs)) {
        match label {
            Some(l) => println!("{}: {}", l, format_tensor(&tensor)),
            None => println!("{}", format_tensor(&tensor)),
        }
    }
}

fn setup() {
    let platform = match (env::consts::OS, env::consts::ARCH) {
        ("macos", "aarch64") => "darwin-arm64",
        ("macos", "x86_64") => "darwin-amd64",
        ("linux", "aarch64") => "linux-arm64",
        ("linux", "x86_64") => "linux-amd64",
        (os, arch) => die(&format!("no prebuilt PJRT CPU plugin for {}-{}", os, arch)),
    };
    let dir = format!("{}/.vector", home());
    fs::create_dir_all(&dir).unwrap_or_else(|e| die(&format!("cannot create {}: {}", dir, e)));
    let url = format!(
        "https://github.com/zml/pjrt-artifacts/releases/latest/download/pjrt-cpu_{}.tar.gz",
        platform
    );
    println!("downloading {}", url);
    let status = Command::new("sh")
        .arg("-c")
        .arg(format!("curl -fL --progress-bar {} | tar xz -C {}", url, dir))
        .status()
        .unwrap_or_else(|e| die(&format!("cannot run curl: {}", e)));
    if !status.success() {
        die("plugin download failed");
    }
    let path = format!("{}/{}", dir, plugin_file());
    if fs::metadata(&path).is_err() {
        die(&format!("download completed but {} is missing", path));
    }
    println!("installed {}", path);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("run") if args.len() == 3 => run(&args[2]),
        Some("build") if args.len() == 3 => print!("{}", compile(&args[2]).0),
        Some("setup") if args.len() == 2 => setup(),
        Some("version") if args.len() == 2 => println!("vector {}", env!("CARGO_PKG_VERSION")),
        Some("help") => println!("{}", USAGE),
        _ => die(USAGE),
    }
}
