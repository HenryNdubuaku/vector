mod algos;
mod audio;
mod batch;
mod builtins;
mod emit;
mod export;
mod grad;
mod graph;
mod image;
mod imports;
mod lexer;
mod linear;
mod net;
mod npy;
mod parser;
mod plot;
mod repl;
mod runtime;
mod safetensors;
mod serve;
mod table;
mod trace;

use std::collections::HashMap;
use std::env;
use std::fs;
use std::process::{exit, Command};

use emit::build_module;
use export::ExportSpec;
use lexer::lex;
use npy::InputSpec;
use parser::Parser;
use plot::FigureSpec;
use runtime::{execute, format_tensor};
use safetensors::SaveSpec;
use trace::{PrintSpec, RowMeta, Tracer};

const USAGE: &str = "usage: vector [file.vec]

  vector                        start the interactive repl
  vector <file.vec>             compile and run a program
  vector serve <m.mlir> [port]  serve an exported model over http (default port 8080)
  vector setup                  detect this machine and install the right backends
  vector version                print version";

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

fn compile(path: &str) -> (String, Vec<InputSpec>, Vec<PrintSpec>, Vec<SaveSpec>, Vec<ExportSpec>, Vec<FigureSpec>, Vec<SaveSpec>) {
    let src = fs::read_to_string(path)
        .unwrap_or_else(|e| die(&format!("cannot read file: {}", e)));
    let lexed = lex(&src);
    let mut p = Parser {
        repl: false,
        library: false,
        toks: lexed.toks,
        cols: lexed.cols,
        lines: lexed.lines,
        pos: 0,
        imports: Vec::new(),
        fns: HashMap::new(),
        modules: HashMap::new(),
    };
    let prog = p.program();
    let (import_fns, import_modules) = imports::load_libraries(path, &prog.imports);
    for name in prog.fns.keys().chain(prog.modules.keys()) {
        if import_fns.contains_key(name) || import_modules.contains_key(name) {
            die(&format!("{} is defined in both an import and {}", name, path));
        }
    }
    let (mut fns, mut modules) = linear::stdlib();
    fns.extend(import_fns);
    fns.extend(prog.fns);
    modules.extend(import_modules);
    modules.extend(prog.modules);
    let mut tracer = Tracer {
        nodes: Vec::new(),
        prints: Vec::new(),
        inputs: Vec::new(),
        saves: Vec::new(),
        exports: Vec::new(),
        figures: Vec::new(),
        figure: FigureSpec::default(),
        plays: Vec::new(),
        loop_prints: Vec::new(),
        modules,
        statics: Vec::new(),
        rng: 0x243F6A8885A308D3,
        claimed: std::collections::HashSet::new(),
        region_depth: 0,
        grad_depth: 0,
        interned: vec![HashMap::new()],
    };
    tracer.trace(&prog.main, &HashMap::new(), &fns);
    if !tracer.figure.series.is_empty() {
        die("plot without savefig or show; finish the figure");
    }
    let mut outputs: Vec<_> = tracer.prints.iter().map(|p| p.val.clone()).collect();
    for spec in &tracer.saves {
        outputs.extend(spec.vals.iter().cloned());
    }
    for spec in &tracer.exports {
        outputs.extend(spec.weight_vals.iter().cloned());
    }
    for fig in &tracer.figures {
        for series in &fig.series {
            outputs.push(series.x.clone());
            outputs.push(series.y.clone());
        }
        outputs.extend(fig.images.iter().cloned());
    }
    for spec in &tracer.plays {
        outputs.extend(spec.vals.iter().cloned());
    }
    let prints = tracer.prints.clone();
    let specs: Vec<InputSpec> = tracer.inputs.iter()
        .map(|(src, id)| match src {
            graph::InputSource::Npy(path) | graph::InputSource::Image(path) | graph::InputSource::Audio(path) => InputSpec {
                path: path.clone(),
                entry: None,
                shape: tracer.nodes[*id].shape.clone(),
                dtype: tracer.nodes[*id].dtype,
            },
            graph::InputSource::Safetensors(path, name) | graph::InputSource::Csv(path, name) => InputSpec {
                path: path.clone(),
                entry: Some(name.clone()),
                shape: tracer.nodes[*id].shape.clone(),
                dtype: tracer.nodes[*id].dtype,
            },
            graph::InputSource::Live(_) => die("internal: live input outside the repl"),
        })
        .collect();
    let params: Vec<usize> = tracer.inputs.iter().map(|&(_, id)| id).collect();
    (build_module(&tracer.nodes, &params, &outputs), specs, prints, tracer.saves, tracer.exports, tracer.figures, tracer.plays)
}

fn print_result(label: &Option<String>, rows: &Option<RowMeta>, tensor: &runtime::Tensor) {
    match rows {
        None => match label {
            Some(l) => println!("{}: {}", l, format_tensor(tensor)),
            None => println!("{}", format_tensor(tensor)),
        },
        Some(rm) => {
            for i in 0..tensor.shape()[0] {
                let tag = format!("{} {}", rm.var, rm.start + i as f64 * rm.step);
                match label {
                    Some(l) => println!("{}: {}: {}", tag, l, tensor.format_row(i)),
                    None => println!("{}: {}", tag, tensor.format_row(i)),
                }
            }
        }
    }
}

fn open_figure(path: &str) {
    let opener = if cfg!(target_os = "macos") { "open" } else { "xdg-open" };
    Command::new(opener)
        .arg(path)
        .spawn()
        .unwrap_or_else(|e| die(&format!("cannot open plot viewer: {}", e)));
}

fn play_audio(path: &str) {
    let player = if cfg!(target_os = "macos") { "afplay" } else { "aplay" };
    Command::new(player)
        .arg(path)
        .spawn()
        .unwrap_or_else(|e| die(&format!("cannot start audio player: {}", e)));
}

fn run(path: &str) {
    let (module, specs, prints, saves, exports, figures, plays) = compile(path);
    let mut results = execute(&module, &specs).into_iter();
    for spec in &prints {
        let tensor = results.next().unwrap();
        print_result(&spec.label, &spec.rows, &tensor);
    }
    for spec in &saves {
        let tensors: Vec<_> = spec.names.iter().map(|_| results.next().unwrap()).collect();
        safetensors::write_save(spec, &tensors);
    }
    for spec in &exports {
        let tensors: Vec<_> = spec.weight_vals.iter().map(|_| results.next().unwrap()).collect();
        export::write_export(spec, &tensors);
    }
    for (i, fig) in figures.iter().enumerate() {
        let count = fig.series.len() * 2 + fig.images.len();
        let tensors: Vec<_> = (0..count).map(|_| results.next().unwrap()).collect();
        let written = plot::write_figure(fig, &tensors, i);
        if fig.path.is_none() {
            open_figure(&written);
        }
    }
    for spec in &plays {
        let tensors: Vec<_> = spec.vals.iter().map(|_| results.next().unwrap()).collect();
        audio::write_wav(spec, &tensors);
        play_audio(&spec.path);
    }
}

fn detect_backends() -> Vec<&'static str> {
    let mut backends = vec!["cpu"];
    match env::consts::OS {
        "macos" => backends.push("metal"),
        "linux" => {
            if fs::metadata("/dev/nvidia0").is_ok() || fs::metadata("/proc/driver/nvidia").is_ok() {
                backends.push("cuda");
            } else if fs::metadata("/dev/kfd").is_ok() {
                backends.push("rocm");
            }
            if fs::metadata("/dev/accel0").is_ok() {
                backends.push("tpu");
            }
        }
        _ => {}
    }
    backends
}

fn setup_auto() {
    let backends = detect_backends();
    println!("detected backends: {}", backends.join(", "));
    for backend in backends {
        if fs::metadata(backend_path(backend)).is_ok() {
            println!("{} already installed", backend);
            continue;
        }
        setup(backend);
    }
}

fn setup(backend: &str) {
    if !matches!(backend, "cpu" | "cuda" | "rocm" | "oneapi" | "tpu" | "metal") {
        die(&format!("unknown backend: {} (expected cpu, cuda, rocm, oneapi, tpu or metal)", backend));
    }
    if backend == "metal" {
        if env::consts::OS != "macos" {
            die("the metal backend needs macos");
        }
    } else if backend != "cpu" && env::consts::OS != "linux" {
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
    } else if backend == "metal" {
        let tag = if env::consts::ARCH == "aarch64" { "arm64" } else { "x86_64" };
        format!(
            "command -v unzip >/dev/null || {{ echo 'vector setup metal needs unzip installed' >&2; exit 1; }}; \
             url=$(curl -fsSL https://pypi.org/simple/jax-metal/ | grep -o 'https://[^\"#]*{tag}\\.whl' | tail -1); \
             [ -n \"$url\" ] || {{ echo 'no jax-metal wheel found on pypi' >&2; exit 1; }}; \
             echo \"downloading $url\" && \
             curl -fL --progress-bar \"$url\" -o {dir}/jaxmetal.whl && \
             unzip -p {dir}/jaxmetal.whl 'jax_plugins/metal_plugin/*.dylib' > {dir}/libpjrt_metal.dylib && \
             rm {dir}/jaxmetal.whl",
            tag = tag,
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
            Some("setup") if args.len() == 2 => setup_auto(),
            Some("setup") if args.len() == 3 => setup(&args[2]),
            Some("serve") if args.len() == 3 => serve::serve(&args[2], 8080),
            Some("serve") if args.len() == 4 => {
                let port = args[3].parse()
                    .unwrap_or_else(|_| die(&format!("port must be a number, got {}", args[3])));
                serve::serve(&args[2], port)
            }
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
