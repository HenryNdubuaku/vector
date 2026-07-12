mod batch;
mod builtins;
mod emit;
mod grad;
mod graph;
mod lexer;
mod linear;
mod npy;
mod parser;
mod repl;
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

const USAGE: &str = "usage: vector [file.vec]

  vector              start the interactive repl
  vector <file.vec>   compile and run a program
  vector setup [b]    download a PJRT plugin to ~/.vector (cpu, cuda, rocm, oneapi, tpu)
  vector version      print version";

struct VectorError(String);

fn die(msg: &str) -> ! {
    std::panic::panic_any(VectorError(msg.to_string()));
}

fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        if let Some(VectorError(msg)) = info.payload().downcast_ref::<VectorError>() {
            eprintln!("{}", msg);
        } else if let Some(msg) = info.payload().downcast_ref::<&str>() {
            eprintln!("internal error: {}", msg);
        } else if let Some(msg) = info.payload().downcast_ref::<String>() {
            eprintln!("internal error: {}", msg);
        } else {
            eprintln!("internal error");
        }
    }));
}

fn home() -> String {
    env::var("HOME").unwrap_or_else(|_| die("HOME is not set"))
}

fn plugin_file(backend: &str) -> String {
    let ext = if cfg!(target_os = "macos") { "dylib" } else { "so" };
    format!("libpjrt_{}.{}", backend, ext)
}

fn backend_path(backend: &str) -> String {
    format!("{}/.vector/{}", home(), plugin_file(backend))
}

fn plugin_path() -> String {
    if let Ok(p) = env::var("PJRT_PLUGIN_PATH") {
        return p;
    }
    if let Ok(backend) = env::var("VECTOR_BACKEND") {
        let path = backend_path(&backend);
        if fs::metadata(&path).is_err() {
            die(&format!("no {} plugin at {}; run `vector setup {}`", backend, path, backend));
        }
        return path;
    }
    for backend in ["tpu", "cuda", "rocm", "oneapi", "cpu"] {
        let path = backend_path(backend);
        if fs::metadata(&path).is_ok() {
            return path;
        }
    }
    die("no PJRT plugin found; run `vector setup` or set PJRT_PLUGIN_PATH");
}

fn compile(path: &str) -> (String, Vec<InputSpec>, Vec<Option<String>>) {
    let src = fs::read_to_string(path)
        .unwrap_or_else(|e| die(&format!("cannot read file: {}", e)));
    let lexed = lex(&src);
    let mut p = Parser {
        repl: false,
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
        .map(|(src, id)| match src {
            graph::InputSource::Npy(path) => InputSpec {
                path: path.clone(),
                shape: tracer.nodes[*id].shape.clone(),
                dtype: tracer.nodes[*id].dtype,
            },
            graph::InputSource::Live(_) => die("internal: live input outside the repl"),
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

fn setup(backend: &str) {
    if !matches!(backend, "cpu" | "cuda" | "rocm" | "oneapi" | "tpu") {
        die(&format!("unknown backend: {} (expected cpu, cuda, rocm, oneapi or tpu)", backend));
    }
    if backend != "cpu" && env::consts::OS != "linux" {
        die(&format!("the {} backend needs linux; only cpu is available on {}", backend, env::consts::OS));
    }
    let platform = match (env::consts::OS, env::consts::ARCH) {
        ("macos", "aarch64") => "darwin-arm64",
        ("macos", "x86_64") => "darwin-amd64",
        ("linux", "aarch64") => "linux-arm64",
        ("linux", "x86_64") => "linux-amd64",
        (os, arch) => die(&format!("no prebuilt PJRT plugin for {}-{}", os, arch)),
    };
    if matches!(backend, "rocm" | "oneapi" | "tpu") && platform != "linux-amd64" {
        die(&format!("the {} plugin is only published for linux-amd64", backend));
    }
    let dir = format!("{}/.vector", home());
    fs::create_dir_all(&dir).unwrap_or_else(|e| die(&format!("cannot create {}: {}", dir, e)));
    let cmd = if backend == "tpu" {
        // google ships libtpu as a wheel (a zip holding libtpu/libtpu.so), not a tarball
        format!(
            "command -v unzip >/dev/null || {{ echo 'vector setup tpu needs unzip installed' >&2; exit 1; }}; \
             url=$(curl -fsSL https://pypi.org/simple/libtpu/ | grep -o 'https://[^\"#]*manylinux[^\"#]*x86_64\\.whl' | tail -1); \
             [ -n \"$url\" ] || {{ echo 'no libtpu wheel found on pypi' >&2; exit 1; }}; \
             echo \"downloading $url\" && \
             curl -fL --progress-bar \"$url\" -o {dir}/libtpu.whl && \
             unzip -p {dir}/libtpu.whl libtpu/libtpu.so > {dir}/libpjrt_tpu.so && \
             rm {dir}/libtpu.whl",
            dir = dir
        )
    } else {
        let url = format!(
            "https://github.com/zml/pjrt-artifacts/releases/latest/download/pjrt-{}_{}.tar.gz",
            backend, platform
        );
        println!("downloading {}", url);
        format!("curl -fL --progress-bar {} | tar xz -C {}", url, dir)
    };
    let status = Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .status()
        .unwrap_or_else(|e| die(&format!("cannot run curl: {}", e)));
    if !status.success() {
        die("plugin download failed");
    }
    let path = format!("{}/{}", dir, plugin_file(backend));
    if fs::metadata(&path).is_err() {
        die(&format!("download completed but {} is missing", path));
    }
    println!("installed {}", path);
}

fn main() {
    install_panic_hook();
    let args: Vec<String> = env::args().collect();
    let outcome = std::panic::catch_unwind(|| {
        match args.get(1).map(String::as_str) {
            Some("setup") if args.len() == 2 => setup("cpu"),
            Some("setup") if args.len() == 3 => setup(&args[2]),
            Some("version") if args.len() == 2 => println!("vector {}", env!("CARGO_PKG_VERSION")),
            Some("help") => println!("{}", USAGE),
            Some(path) if args.len() == 2 => run(path),
            None => repl::run_repl(),
            _ => die(USAGE),
        }
    });
    if outcome.is_err() {
        exit(1);
    }
}
