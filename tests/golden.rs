use std::fs;
use std::process::Command;

fn run_vector(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_vector")).args(args).output().unwrap()
}

fn npy_bytes(descr: &str, shape: &str, data: &[u8]) -> Vec<u8> {
    let mut header = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}", descr, shape);
    let pad = (64 - (10 + header.len() + 1) % 64) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend((header.len() as u16).to_le_bytes());
    bytes.extend(header.as_bytes());
    bytes.extend(data);
    bytes
}

fn write_fixtures() {
    fs::create_dir_all("tests/cases/data").unwrap();
    let m: Vec<u8> = [1.5f32, 2.5, 3.5, 4.5].iter().flat_map(|x| x.to_le_bytes()).collect();
    fs::write("tests/cases/data/m.npy", npy_bytes("<f4", "(2, 2)", &m)).unwrap();
    let v: Vec<u8> = [3.0f64, 4.0].iter().flat_map(|x| x.to_le_bytes()).collect();
    fs::write("tests/cases/data/v.npy", npy_bytes("<f8", "(2,)", &v)).unwrap();
}

#[test]
fn golden_cases() {
    write_fixtures();
    let mut ran = 0;
    for entry in fs::read_dir("tests/cases").unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("vec") {
            continue;
        }
        let expected = fs::read_to_string(path.with_extension("out")).unwrap();
        let output = run_vector(&[path.to_str().unwrap()]);
        let stdout = String::from_utf8(output.stdout).unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(output.status.success(), "{}: {}", path.display(), stderr);
        assert_eq!(stdout, expected, "{}", path.display());
        ran += 1;
    }
    assert!(ran >= 7);
}

#[test]
fn shape_mismatch_fails_at_trace_time() {
    let path = std::env::temp_dir().join("vector_shape_mismatch.vec");
    fs::write(&path, "print([1.0, 2.0] + [1.0, 2.0, 3.0])\n").unwrap();
    let output = run_vector(&[path.to_str().unwrap()]);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success());
    assert!(stderr.contains("[2]") && stderr.contains("[3]"), "{}", stderr);
}

#[test]
fn size_one_stretching_is_rejected() {
    let path = std::env::temp_dir().join("vector_stretch.vec");
    fs::write(&path, "print([[1.0], [2.0]] * [[1.0, 2.0], [3.0, 4.0]])\n").unwrap();
    let output = run_vector(&[path.to_str().unwrap()]);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success());
    assert!(stderr.contains("[2, 1]") && stderr.contains("[2, 2]"), "{}", stderr);
}

#[test]
fn grad_requires_scalar_output() {
    let path = std::env::temp_dir().join("vector_grad_nonscalar.vec");
    fs::write(&path, "fn f(x):\n  x * 2.0\n\nprint(grad(f, [1.0, 2.0]))\n").unwrap();
    let output = run_vector(&[path.to_str().unwrap()]);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success());
    assert!(stderr.contains("scalar") && stderr.contains("[2]"), "{}", stderr);
}

#[test]
fn load_missing_file_fails_loud() {
    let path = std::env::temp_dir().join("vector_load_missing.vec");
    fs::write(&path, "print(load(\"/nonexistent/data.npy\"))\n").unwrap();
    let output = run_vector(&[path.to_str().unwrap()]);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success());
    assert!(stderr.contains("/nonexistent/data.npy"), "{}", stderr);
}

#[test]
fn mixed_depth_vmap_args_fail_loud() {
    let path = std::env::temp_dir().join("vector_mixed_vmap.vec");
    fs::write(
        &path,
        "fn f(a, b):\n  a * b\n\nfn g(r):\n  vmap(f, r, [1.0, 2.0])\n\nprint(vmap(g, [[1.0, 2.0], [3.0, 4.0]]))\n",
    )
    .unwrap();
    let output = run_vector(&[path.to_str().unwrap()]);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success());
    assert!(stderr.contains("batching depth"), "{}", stderr);
}

#[test]
fn print_inside_for_fails_loud() {
    let path = std::env::temp_dir().join("vector_print_in_for.vec");
    fs::write(&path, "w = 1.0\nfor i in 0..3:\n  w = w * 2.0\n  print(w)\nprint(w)\n").unwrap();
    let output = run_vector(&[path.to_str().unwrap()]);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success());
    assert!(stderr.contains("print inside a for loop"), "{}", stderr);
}


#[test]
fn record_field_mismatch_fails_loud() {
    let path = std::env::temp_dir().join("vector_record_mismatch.vec");
    fs::write(&path, "print({a: 1.0, b: 2.0} + {a: 1.0, c: 2.0})\n").unwrap();
    let output = run_vector(&[path.to_str().unwrap()]);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success());
    assert!(stderr.contains("record fields mismatch"), "{}", stderr);
}

#[test]
fn matmul_on_record_fails_loud() {
    let path = std::env::temp_dir().join("vector_record_matmul.vec");
    fs::write(&path, "p = {a: [[1.0]]}\nprint(matmul(p, p))\n").unwrap();
    let output = run_vector(&[path.to_str().unwrap()]);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success());
    assert!(stderr.contains("cannot be a record"), "{}", stderr);
}

#[test]
fn matmul_contraction_mismatch_fails() {
    let path = std::env::temp_dir().join("vector_matmul_mismatch.vec");
    fs::write(&path, "print(matmul([[1.0, 2.0]], [[1.0, 2.0]]))\n").unwrap();
    let output = run_vector(&[path.to_str().unwrap()]);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success());
    assert!(stderr.contains("matmul"), "{}", stderr);
}

fn run_vector_src(name: &str, src: &str) -> std::process::Output {
    let path = std::env::temp_dir().join(name);
    fs::write(&path, src).unwrap();
    run_vector(&[path.to_str().unwrap()])
}

const SCALE_MODULE: &str = "module Scale(k):
  s = 0.0 + k
  forward(self, x):
    self.s * x
  loss(self, t):
    d = self([1.0, 2.0]) - t
    sum(d * d)

";

#[test]
fn save_and_load_module_roundtrip() {
    fs::create_dir_all("tests/cases/data").unwrap();
    let a = run_vector_src("vector_save_a.vec", &format!(
        "{}m = Scale(4)\nfor i in 0..5:\n  m = m - 0.05 * grad(m.loss, [3.0, 6.0])\nsave(m, \"tests/cases/data/scale.safetensors\")\nprint(m.loss([3.0, 6.0]))\nprint(m.s)\n",
        SCALE_MODULE));
    assert!(a.status.success(), "{}", String::from_utf8_lossy(&a.stderr));
    let a_out = String::from_utf8(a.stdout).unwrap();
    let b = run_vector_src("vector_save_b.vec", &format!(
        "{}m = load(\"tests/cases/data/scale.safetensors\")\nprint(m.loss([3.0, 6.0]))\nprint(m.s)\nm2 = m - 0.05 * grad(m.loss, [3.0, 6.0])\nprint(m2.s)\n",
        SCALE_MODULE));
    assert!(b.status.success(), "{}", String::from_utf8_lossy(&b.stderr));
    let b_out = String::from_utf8(b.stdout).unwrap();
    let a_lines: Vec<&str> = a_out.lines().collect();
    let b_lines: Vec<&str> = b_out.lines().collect();
    assert_eq!(a_lines[0], b_lines[0], "loss differs after reload");
    assert_eq!(a_lines[1], b_lines[1], "weights differ after reload");
    assert_ne!(b_lines[1], b_lines[2], "training from the checkpoint made no progress");
}

#[test]
fn save_npy_roundtrip() {
    fs::create_dir_all("tests/cases/data").unwrap();
    let a = run_vector_src("vector_npy_a.vec",
        "x = [1.0, 2.0, 3.0] * 2.0\nsave(x, \"tests/cases/data/doubled.npy\")\nprint(x)\n");
    assert!(a.status.success(), "{}", String::from_utf8_lossy(&a.stderr));
    let b = run_vector_src("vector_npy_b.vec",
        "print(load(\"tests/cases/data/doubled.npy\") + 1.0)\n");
    assert!(b.status.success(), "{}", String::from_utf8_lossy(&b.stderr));
    assert_eq!(String::from_utf8(b.stdout).unwrap(), "[3, 5, 7] : f32\n");
}

#[test]
fn load_pytorch_style_names() {
    fs::create_dir_all("tests/cases/data").unwrap();
    let mut header = String::from(
        "{\"layers.0.weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},\"scale\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[8,12]}}");
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.as_bytes());
    for x in [1.5f32, 2.5, 3.0] {
        bytes.extend(x.to_le_bytes());
    }
    fs::write("tests/cases/data/foreign.safetensors", bytes).unwrap();
    let out = run_vector_src("vector_foreign.vec",
        "p = load(\"tests/cases/data/foreign.safetensors\")\nprint(p.layers._0.weight * p.scale)\n");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "[4.5, 7.5] : f32\n");
}

#[test]
fn load_after_save_in_same_program() {
    fs::create_dir_all("tests/cases/data").unwrap();
    let _ = fs::remove_file("tests/cases/data/samerun.safetensors");
    let out = run_vector_src("vector_same_run.vec",
        "x = [1.0, 2.0]\nsave({v: x * 3.0}, \"tests/cases/data/samerun.safetensors\")\np = load(\"tests/cases/data/samerun.safetensors\")\nprint(p.v)\n");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "[3, 6] : f32\n");
    assert!(fs::metadata("tests/cases/data/samerun.safetensors").is_ok(), "checkpoint was not written");
}

#[test]
fn save_inside_for_fails_loud() {
    let out = run_vector_src("vector_save_in_for.vec",
        "w = 1.0\nfor i in 0..2:\n  w = w * 2.0\n  save(w, \"tests/cases/data/w.npy\")\nprint(w)\n");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(!out.status.success());
    assert!(stderr.contains("save inside a for loop"), "{}", stderr);
}

#[test]
fn save_record_to_npy_fails_loud() {
    let out = run_vector_src("vector_save_record_npy.vec",
        "save({a: 1.0}, \"tests/cases/data/bad.npy\")\n");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(!out.status.success());
    assert!(stderr.contains(".safetensors"), "{}", stderr);
}

#[test]
fn load_undefined_module_fails_loud() {
    fs::create_dir_all("tests/cases/data").unwrap();
    let a = run_vector_src("vector_orphan_a.vec",
        "module Osc(k):\n  c = 0.0 + k\n  forward(self, x):\n    self.c * x\n\no = Osc(2)\nsave(o, \"tests/cases/data/osc.safetensors\")\nprint(o.c)\n");
    assert!(a.status.success(), "{}", String::from_utf8_lossy(&a.stderr));
    let b = run_vector_src("vector_orphan_b.vec",
        "m = load(\"tests/cases/data/osc.safetensors\")\nprint(m.c)\n");
    let stderr = String::from_utf8(b.stderr).unwrap();
    assert!(!b.status.success());
    assert!(stderr.contains("define module Osc"), "{}", stderr);
}

fn run_repl_script(script: &str) -> (String, String) {
    use std::io::Write;
    use std::process::Stdio;
    let mut child = Command::new(env!("CARGO_BIN_EXE_vector"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(script.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    (String::from_utf8(out.stdout).unwrap(), String::from_utf8(out.stderr).unwrap())
}

#[test]
fn repl_persists_state_and_recovers_from_errors() {
    let (stdout, stderr) = run_repl_script(
        "x = 2.0\nx * 3.0\ny = [1.0, 2.0] + x\nnope_undefined\ny * y\n",
    );
    assert_eq!(stdout, "6 : f32\n[9, 16] : f32\n", "stderr: {}", stderr);
    assert!(stderr.contains("undefined: nope_undefined"), "{}", stderr);
}

#[test]
fn repl_trains_across_chunks() {
    let script = "\
fn loss(w):
  d = w - [3.0, 4.0]
  mean(d * d)

w = [0.0, 0.0]
for i in 0..10:
  w = w - 1.0 * grad(loss, w)

w
";
    let (stdout, stderr) = run_repl_script(script);
    assert_eq!(stdout, "[3, 4] : f32\n", "stderr: {}", stderr);
}

#[test]
fn repl_saves_and_loads() {
    fs::create_dir_all("tests/cases/data").unwrap();
    let (stdout, stderr) = run_repl_script(
        "x = [1.0, 2.0]\nq = save({v: x * 2.0}, \"tests/cases/data/repl.safetensors\")\np = load(\"tests/cases/data/repl.safetensors\")\np.v\n",
    );
    assert_eq!(stdout, "[2, 4] : f32\n", "stderr: {}", stderr);
}

#[test]
fn repl_redefines_functions() {
    let (stdout, _) = run_repl_script(
        "fn f(x):\n  x * 2.0\n\nsum(f(3.0))\nfn f(x):\n  x * 10.0\n\nsum(f(3.0))\n",
    );
    assert_eq!(stdout, "6 : f32\n30 : f32\n");
}
