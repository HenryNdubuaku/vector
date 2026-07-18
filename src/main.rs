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
mod stdlib;
mod net;
mod npy;
mod parser;
mod plot;
mod repl;
mod runtime;
mod safetensors;
mod serve;
mod table;
mod text;
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
  vector test [steps]           train a small model on the cpu and the accelerator to check the install
  vector benchmark [steps]      the same run, benchmark-sized (200 steps), with the config printed
  vector version                print version

  --accelerate                  run on the machine's accelerator (gpu/tpu); programs run on the cpu by default

  set VECTOR_LOGS=1 to see the XLA runtime logs vector hides by default";

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

fn installed_accelerator() -> Option<&'static str> {
    ["tpu", "cuda", "rocm", "oneapi", "metal"].into_iter()
        .find(|b| fs::metadata(backend_path(b)).is_ok())
}

static FLAG_BACKEND: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();

fn engage_accelerator() {
    let found = installed_accelerator()
        .unwrap_or_else(|| die("no accelerator installed; run `vector setup`"));
    let _ = FLAG_BACKEND.set(found);
    eprintln!("accelerating on {}", found);
}

fn plugin_path() -> String {
    if let Ok(p) = env::var("PJRT_PLUGIN_PATH") {
        return p;
    }
    if let Some(backend) = FLAG_BACKEND.get() {
        return backend_path(backend);
    }
    if let Ok(backend) = env::var("VECTOR_BACKEND") {
        let path = backend_path(&backend);
        if fs::metadata(&path).is_err() {
            die(&format!("no {} plugin at {}; run `vector setup {}`", backend, path, backend));
        }
        return path;
    }
    let path = backend_path("cpu");
    if fs::metadata(&path).is_ok() {
        return path;
    }
    die("no PJRT plugin found; run `vector setup` or set PJRT_PLUGIN_PATH");
}

fn compile(path: &str) -> (String, Vec<Option<InputSpec>>, Vec<PrintSpec>, Vec<SaveSpec>, Vec<ExportSpec>, Vec<FigureSpec>, Vec<SaveSpec>) {
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
    let (mut fns, mut modules) = stdlib::stdlib();
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
        decodes: HashMap::new(),
        modules,
        statics: Vec::new(),
        rng: 0x243F6A8885A308D3,
        rng_sites: 0,
        rng_baked: false,
        seed: None,
        loop_counters: Vec::new(),
        while_depth: 0,
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
    let specs: Vec<Option<InputSpec>> = tracer.inputs.iter()
        .map(|(src, id)| match src {
            graph::InputSource::Npy(path) | graph::InputSource::Image(path) | graph::InputSource::Audio(path) | graph::InputSource::Text(path) => Some(InputSpec {
                path: path.clone(),
                entry: None,
                shape: tracer.nodes[*id].shape.clone(),
                dtype: tracer.nodes[*id].dtype,
            }),
            graph::InputSource::Tokens(path, name) | graph::InputSource::Safetensors(path, name) | graph::InputSource::Csv(path, name) => Some(InputSpec {
                path: path.clone(),
                entry: Some(name.clone()),
                shape: tracer.nodes[*id].shape.clone(),
                dtype: tracer.nodes[*id].dtype,
            }),
            graph::InputSource::Seed => None,
            graph::InputSource::Live(_) => die("internal: live input outside the repl"),
        })
        .collect();
    let params: Vec<usize> = tracer.inputs.iter().map(|&(_, id)| id).collect();
    let module = build_module(&tracer.nodes, &params, &outputs);
    if let Ok(dump) = env::var("VECTOR_DUMP_MLIR") {
        fs::write(&dump, &module).unwrap_or_else(|e| die(&format!("cannot write {}: {}", dump, e)));
    }
    (module, specs, prints, tracer.saves, tracer.exports, tracer.figures, tracer.plays)
}

fn render(decode: &trace::Decode, vals: &[f64]) -> String {
    match decode {
        trace::Decode::Bytes => text::bytes_to_string(vals),
        trace::Decode::Tokens(tok) => text::decode_ids(vals, tok),
    }
}

fn print_result(label: &Option<String>, rows: &Option<RowMeta>, decode: &Option<trace::Decode>, tensor: &runtime::Tensor) {
    let body = |vals: &[f64], plain: String| match decode {
        Some(d) => render(d, vals),
        None => plain,
    };
    match rows {
        None => {
            let text = body(&tensor.f64_vec(), format_tensor(tensor));
            match label {
                Some(l) => println!("{}: {}", l, text),
                None => println!("{}", text),
            }
        }
        Some(rm) => {
            let all = tensor.f64_vec();
            let row_len: usize = tensor.shape()[1..].iter().product::<usize>().max(1);
            for i in 0..tensor.shape()[0] {
                let tag = format!("{} {}", rm.var, rm.start + i as f64 * rm.step);
                let text = body(&all[i * row_len..(i + 1) * row_len], tensor.format_row(i));
                match label {
                    Some(l) => println!("{}: {}: {}", tag, l, text),
                    None => println!("{}: {}", tag, text),
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

const TEST_SOURCE: &str = "
inputs = reshape(linspace(-pi, pi, 2048), 2048, 1)
targets = sin(inputs)

module TestNet(hidden, hsq):
  w1 = reshape(sin(arange(hidden)), 1, hidden)
  b1 = zeros(hidden)
  w2 = reshape(sin(arange(hsq)), hidden, hidden) * 0.03
  b2 = zeros(hidden)
  w3 = reshape(sin(arange(hidden)), hidden, 1) * 0.05

  forward(self, x):
    h1 = tanh(matmul(x, self.w1) + self.b1)
    h2 = tanh(matmul(h1, self.w2) + self.b2)
    matmul(h2, self.w3)

  loss(self, x, t):
    d = self(x) - t
    mean(d * d)

m = TestNet(1024, 1048576)
before = m.loss(inputs, targets)
for i in 0..STEPS:
  m = m - 0.001 * grad(m.loss, inputs, targets)
print(before)
print(m.loss(inputs, targets))
";

fn self_test(steps: usize) {
    let source = TEST_SOURCE.replace("STEPS", &steps.to_string());
    let path = env::temp_dir().join("vector_self_test.vec");
    fs::write(&path, source).unwrap_or_else(|e| die(&format!("cannot write self test: {}", e)));
    let (module, _, _, _, _, _, _) = compile(path.to_str().unwrap());
    let mut backends = vec!["cpu"];
    backends.extend(installed_accelerator());
    if backends.contains(&"metal") {
        runtime::disable_muffling();
    }
    let mut cpu_seconds = 0.0;
    for backend in backends {
        let attempt = || {
            let start = std::time::Instant::now();
            let engine = runtime::Engine::with_path(backend_path(backend));
            let executable = engine.prepare(&module);
            let compiled = start.elapsed();
            let results = engine.run(&executable, Vec::new());
            let mut times: Vec<f64> = (0..5)
                .map(|_| {
                    let start = std::time::Instant::now();
                    engine.run(&executable, Vec::new());
                    start.elapsed().as_secs_f64()
                })
                .collect();
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let ran = times[2];
            let before = results[0].f64_vec()[0];
            let after = results[1].f64_vec()[0];
            if !after.is_finite() || after >= before {
                die(&format!("{}: training failed (loss {} -> {})", backend, before, after));
            }
            (before, after, compiled.as_secs_f64(), ran)
        };
        let mut outcome = std::panic::catch_unwind(attempt);
        for _ in 0..2 {
            if outcome.is_ok() {
                break;
            }
            if backend != "metal" {
                break;
            }
            eprintln!("retrying metal (its compiler occasionally fails; an Apple plugin bug)");
            outcome = std::panic::catch_unwind(attempt);
        }
        let Ok((before, after, compiled, ran)) = outcome else {
            exit(1);
        };
        let speedup = if backend == "cpu" {
            cpu_seconds = ran;
            String::new()
        } else {
            format!(", {:.1}x the cpu", cpu_seconds / ran)
        };
        println!(
            "{}: ok — loss {:.4} -> {:.4} (compiled in {:.2}s, trained in {:.2}s median of 5{})",
            backend, before, after, compiled, ran, speedup
        );
    }
    if installed_accelerator().is_none() {
        println!("no accelerator installed; tested the cpu only");
    }
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
        print_result(&spec.label, &spec.rows, &spec.decode, &tensor);
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
            if fs::metadata("/dev/accel0").is_ok() || google_pci_device() {
                backends.push("tpu");
            }
        }
        _ => {}
    }
    backends
}

fn google_pci_device() -> bool {
    let Ok(entries) = fs::read_dir("/sys/bus/pci/devices") else {
        return false;
    };
    entries.flatten().any(|e| {
        fs::read_to_string(e.path().join("vendor"))
            .map(|v| v.trim() == "0x1ae0")
            .unwrap_or(false)
    })
}

fn setup_auto() {
    let backends = detect_backends();
    println!("detected backends: {}", backends.join(", "));
    for &backend in &backends {
        if fs::metadata(backend_path(backend)).is_ok() {
            println!("{} already installed", backend);
            continue;
        }
        setup(backend);
    }
    report_default();
}

fn report_default() {
    if fs::metadata(backend_path("cpu")).is_ok() {
        println!("programs run on the cpu; add --accelerate to use the gpu/tpu");
    } else {
        println!("no cpu plugin installed; run `vector setup`");
    }
    if let Some(b) = installed_accelerator() {
        println!("--accelerate will run on: {}", b);
        let missing = runtime::missing_libs(&backend_path(b));
        if !missing.is_empty() {
            println!("warning: {} won't load until its libraries are installed:{}", b, missing);
        } else if b == "cuda" && !libdevice_present() {
            println!("warning: xla also needs libdevice (part of nvcc) to compile for the gpu; {}", runtime::CUDA_RECIPE);
        }
    }
}

fn libdevice_present() -> bool {
    if env::var("XLA_FLAGS").map(|f| f.contains("xla_gpu_cuda_data_dir")).unwrap_or(false) {
        return true;
    }
    let mut roots = vec![
        "/usr/local/cuda".to_string(),
        "/opt/cuda".to_string(),
        format!("{}/.vector/cuda", home()),
    ];
    if let Ok(entries) = fs::read_dir("/usr/local") {
        for entry in entries.flatten() {
            if entry.file_name().to_string_lossy().starts_with("cuda-") {
                roots.push(entry.path().to_string_lossy().into_owned());
            }
        }
    }
    roots.iter().any(|r| fs::metadata(format!("{}/nvvm/libdevice/libdevice.10.bc", r)).is_ok())
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
    let missing = runtime::missing_libs(&path);
    if !missing.is_empty() {
        println!("warning: the {} plugin needs libraries this machine doesn't have yet:{}", backend, missing);
    }
}

fn main() {
    install_panic_hook();
    if env::var_os("VECTOR_LOGS").is_none() && env::var_os("TF_CPP_MIN_LOG_LEVEL").is_none() {
        unsafe { env::set_var("TF_CPP_MIN_LOG_LEVEL", "2") };
    }
    let mut args: Vec<String> = env::args().collect();
    let accelerate = args.iter().any(|a| a == "--accelerate");
    args.retain(|a| a != "--accelerate");
    let outcome = std::panic::catch_unwind(|| {
        if accelerate {
            engage_accelerator();
        }
        match args.get(1).map(String::as_str) {
            Some("setup") if args.len() == 2 => setup_auto(),
            Some("setup") if args.len() == 3 => {
                setup(&args[2]);
                report_default();
            }
            Some("test") if args.len() == 2 => self_test(20),
            Some("test") if args.len() == 3 => {
                let steps = args[2].parse()
                    .unwrap_or_else(|_| die(&format!("steps must be a number, got {}", args[2])));
                self_test(steps)
            }
            Some("benchmark") if args.len() == 2 || args.len() == 3 => {
                let steps = match args.get(2) {
                    Some(s) => s.parse()
                        .unwrap_or_else(|_| die(&format!("steps must be a number, got {}", s))),
                    None => 200,
                };
                println!("{} full-batch steps of a 1-1024-1024-1 tanh net on 2048 points of sin(x), f32", steps);
                self_test(steps)
            }
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
    std::mem::forget(runtime::exit_muffle());
    if outcome.is_err() {
        exit(1);
    }
}
