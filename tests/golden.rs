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
fn self_test_passes_on_this_machine() {
    let output = run_vector(&["test"]);
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert!(stdout.contains("cpu: ok"), "{}", stdout);
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
fn print_inside_nested_loops_fails_loud() {
    let output = run_vector_src("vector_print_nested.vec",
        "fn inner(x):\n  y = x\n  for j in 0..2:\n    y = y + 1.0\n    print(y)\n  y\n\nw = 0.0\nfor i in 0..2:\n  w = inner(w)\nprint(w)\n");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success());
    assert!(stderr.contains("nested loops"), "{}", stderr);
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
fn export_writes_standalone_stablehlo() {
    fs::create_dir_all("tests/cases/data").unwrap();
    let out = run_vector_src("vector_export.vec", &format!(
        "{}m = Scale(4)\nfor i in 0..5:\n  m = m - 0.05 * grad(m.loss, [3.0, 6.0])\nexport(m, \"tests/cases/data/scale.mlir\", [1.0, 2.0])\nprint(m.s)\n",
        SCALE_MODULE));
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "3.03125 : f32\n");
    let mlir = fs::read_to_string("tests/cases/data/scale.mlir").unwrap();
    assert!(mlir.contains("func.func @main(%"), "{}", mlir);
    assert!(mlir.contains("tensor<2xf32>"), "{}", mlir);
    assert!(mlir.contains("stablehlo.multiply"), "{}", mlir);
    assert!(mlir.contains("stablehlo.constant dense<3.03125>"), "{}", mlir);
}

#[test]
fn csv_loads_columns_and_factorizes() {
    fs::create_dir_all("tests/cases/data").unwrap();
    fs::write("tests/cases/data/mini.csv",
        "sepal length,species,note\n5.1,setosa,\"a, b\"\n4.9,versicolor,c\n4.7,setosa,c\n").unwrap();
    let out = run_vector_src("vector_csv_load.vec",
        "t = load(\"tests/cases/data/mini.csv\")\nprint(t.sepal_length)\nprint(t.species)\nprint(t.note)\n");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        "[5.1, 4.9, 4.7] : f32\n[0, 1, 0] : f32\n[0, 1, 1] : f32\n"
    );
}

#[test]
fn csv_save_and_load_roundtrip() {
    fs::create_dir_all("tests/cases/data").unwrap();
    let a = run_vector_src("vector_csv_a.vec",
        "save({a: [1.0, 2.0], b: [3.5, 4.5]}, \"tests/cases/data/out.csv\")\nprint(1.0)\n");
    assert!(a.status.success(), "{}", String::from_utf8_lossy(&a.stderr));
    assert_eq!(fs::read_to_string("tests/cases/data/out.csv").unwrap(), "a,b\n1,3.5\n2,4.5\n");
    let b = run_vector_src("vector_csv_b.vec",
        "t = load(\"tests/cases/data/out.csv\")\nprint(t.b)\nprint(transpose([t.a, t.b]))\n");
    assert!(b.status.success(), "{}", String::from_utf8_lossy(&b.stderr));
    assert_eq!(
        String::from_utf8(b.stdout).unwrap(),
        "[3.5, 4.5] : f32\n[[1, 3.5], [2, 4.5]] : f32\n"
    );
}

#[test]
fn csv_empty_cell_fails_loud() {
    fs::create_dir_all("tests/cases/data").unwrap();
    fs::write("tests/cases/data/holes.csv", "a,b\n1,\n2,3\n").unwrap();
    let out = run_vector_src("vector_csv_holes.vec",
        "t = load(\"tests/cases/data/holes.csv\")\nprint(t.a)\n");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(!out.status.success());
    assert!(stderr.contains("row 2, column b is empty"), "{}", stderr);
}

#[test]
fn png_load_decodes_reference_files() {
    let out = run_vector_src("vector_png_load.vec",
        "g = load(\"tests/fixtures/gray.png\")\nprint(g * 255.0)\np = load(\"tests/fixtures/pal.png\")\nprint(p)\nf = load(\"tests/fixtures/filtered.png\")\nprint(f * 255.0)\nimg = load(\"tests/fixtures/gradient.png\")\nprint(crop(img, 0, 0, 1, 2) * 255.0)\n");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        "[[0, 64], [128, 255]] : f32\n\
         [[[1, 0, 0], [0, 0, 1]]] : f32\n\
         [[10, 30], [15, 40]] : f32\n\
         [[[0, 0, 0], [8, 0, 4]]] : f32\n"
    );
}

#[test]
fn png_save_roundtrip() {
    fs::create_dir_all("tests/cases/data").unwrap();
    let a = run_vector_src("vector_png_a.vec",
        "img = [[0.0, 0.2], [0.8, 1.0]] * 1.0\nsave(img, \"tests/cases/data/rt.png\")\nprint(img)\n");
    assert!(a.status.success(), "{}", String::from_utf8_lossy(&a.stderr));
    let b = run_vector_src("vector_png_b.vec",
        "print(load(\"tests/cases/data/rt.png\") * 255.0)\n");
    assert!(b.status.success(), "{}", String::from_utf8_lossy(&b.stderr));
    assert_eq!(String::from_utf8(b.stdout).unwrap(), "[[0, 51], [204, 255]] : f32\n");
}

#[test]
fn imshow_embeds_png_in_svg() {
    fs::create_dir_all("tests/cases/data").unwrap();
    let out = run_vector_src("vector_imshow.vec",
        "imshow(load(\"tests/fixtures/gradient.png\"))\ntitle(\"gradient\")\nsavefig(\"tests/cases/data/img.svg\")\nprint(1.0)\n");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let svg = fs::read_to_string("tests/cases/data/img.svg").unwrap();
    assert!(svg.contains("<image "), "{}", svg);
    assert!(svg.contains("data:image/png;base64,"), "{}", svg);
    assert!(svg.contains(">gradient</text>"), "{}", svg);
}

fn write_temp(name: &str, src: &str) {
    fs::write(std::env::temp_dir().join(name), src).unwrap();
}

#[test]
fn import_shares_module_across_programs() {
    fs::create_dir_all("tests/cases/data").unwrap();
    write_temp("vector_lib_scale.vec", SCALE_MODULE);
    let a = run_vector_src("vector_imp_train.vec",
        "import vector_lib_scale\n\nm = Scale(4)\nfor i in 0..5:\n  m = m - 0.05 * grad(m.loss, [3.0, 6.0])\nsave(m, \"tests/cases/data/imp.safetensors\")\nprint(m.s)\n");
    assert!(a.status.success(), "{}", String::from_utf8_lossy(&a.stderr));
    let b = run_vector_src("vector_imp_infer.vec",
        "import vector_lib_scale\n\nm = load(\"tests/cases/data/imp.safetensors\")\nprint(m.s)\nprint(m([2.0]))\n");
    assert!(b.status.success(), "{}", String::from_utf8_lossy(&b.stderr));
    assert_eq!(String::from_utf8(b.stdout).unwrap(), "3.03125 : f32\n[6.0625] : f32\n");
}

#[test]
fn import_resolves_transitively() {
    write_temp("vector_lib_a.vec", "import vector_lib_b\n\nfn double_sq(x):\n  sq(x) * 2.0\n");
    write_temp("vector_lib_b.vec", "fn sq(x):\n  x * x\n");
    let out = run_vector_src("vector_imp_trans.vec",
        "import vector_lib_a\n\nprint(double_sq(3.0))\nprint(sq(4.0))\n");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "18 : f32\n16 : f32\n");
}

#[test]
fn circular_import_fails_loud() {
    write_temp("vector_lib_c1.vec", "import vector_lib_c2\n\nfn f(x):\n  x\n");
    write_temp("vector_lib_c2.vec", "import vector_lib_c1\n\nfn g(x):\n  x\n");
    let out = run_vector_src("vector_imp_circ.vec", "import vector_lib_c1\n\nprint(f(1.0))\n");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(!out.status.success());
    assert!(stderr.contains("circular import"), "{}", stderr);
}

#[test]
fn import_with_top_level_code_fails_loud() {
    write_temp("vector_lib_body.vec", "fn f(x):\n  x\n\nprint(f(1.0))\n");
    let out = run_vector_src("vector_imp_body.vec", "import vector_lib_body\n\nprint(f(1.0))\n");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(!out.status.success());
    assert!(stderr.contains("top-level code"), "{}", stderr);
}

#[test]
fn import_name_collision_fails_loud() {
    write_temp("vector_lib_d1.vec", "fn same(x):\n  x\n");
    write_temp("vector_lib_d2.vec", "fn same(x):\n  x * 2.0\n");
    let out = run_vector_src("vector_imp_coll.vec",
        "import vector_lib_d1\nimport vector_lib_d2\n\nprint(same(1.0))\n");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(!out.status.success());
    assert!(stderr.contains("more than one imported file"), "{}", stderr);
}

#[test]
fn missing_import_fails_loud() {
    let out = run_vector_src("vector_imp_missing.vec", "import vector_nope\n\nprint(1.0)\n");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(!out.status.success());
    assert!(stderr.contains("cannot read import"), "{}", stderr);
}

fn http_request(port: u16, method: &str, body: &str) -> String {
    use std::io::{Read, Write};
    let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(
        s,
        "{} / HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        method, body.len(), body
    ).unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    out
}

fn wait_for_port(port: u16) {
    for _ in 0..100 {
        if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    panic!("server on port {} did not start", port);
}

#[test]
fn serve_answers_http_inference() {
    fs::create_dir_all("tests/cases/data").unwrap();
    let a = run_vector_src("vector_serve_model.vec", &format!(
        "{}m = Scale(3)\nexport(m, \"tests/cases/data/serve.mlir\", [1.0, 2.0])\nprint(m.s)\n",
        SCALE_MODULE));
    assert!(a.status.success(), "{}", String::from_utf8_lossy(&a.stderr));
    let mut server = Command::new(env!("CARGO_BIN_EXE_vector"))
        .args(["serve", "tests/cases/data/serve.mlir", "8643"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    wait_for_port(8643);
    let signature = http_request(8643, "GET", "");
    let good = http_request(8643, "POST", "{\"inputs\": [[2.0, 4.0]]}");
    let bad = http_request(8643, "POST", "{\"inputs\": [[2.0]]}");
    server.kill().ok();
    server.wait().ok();
    assert!(signature.contains("{\"inputs\":[\"2xf32\"],\"outputs\":[\"2xf32\"]}"), "{}", signature);
    assert!(good.contains("{\"outputs\":[[6,12]]}"), "{}", good);
    assert!(bad.contains("400"), "{}", bad);
    assert!(bad.contains("\"error\""), "{}", bad);
}

#[test]
fn load_from_url_downloads_and_caches() {
    fs::create_dir_all("tests/cases/data").unwrap();
    let d: Vec<u8> = [7.0f32, 9.0].iter().flat_map(|x| x.to_le_bytes()).collect();
    fs::write("tests/cases/data/url_data.npy", npy_bytes("<f4", "(2,)", &d)).unwrap();
    let mut server = Command::new("python3")
        .args(["-m", "http.server", "8641", "--bind", "127.0.0.1", "--directory", "tests/cases/data"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    for _ in 0..50 {
        if std::net::TcpStream::connect(("127.0.0.1", 8641)).is_ok() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let url = format!("http://127.0.0.1:8641/url_data.npy?v={}", std::process::id());
    let src = format!("print(load(\"{}\") * 2.0)\n", url);
    let first = run_vector_src("vector_url_a.vec", &src);
    let first_ok = first.status.success();
    let first_out = String::from_utf8(first.stdout).unwrap();
    let first_err = String::from_utf8_lossy(&first.stderr).to_string();
    server.kill().ok();
    server.wait().ok();
    assert!(first_ok, "{}", first_err);
    assert_eq!(first_out, "[14, 18] : f32\n");
    let second = run_vector_src("vector_url_b.vec", &src);
    assert!(second.status.success(), "cache miss after server death: {}", String::from_utf8_lossy(&second.stderr));
    assert_eq!(String::from_utf8(second.stdout).unwrap(), "[14, 18] : f32\n");
}

#[test]
fn save_to_url_fails_loud() {
    let out = run_vector_src("vector_url_save.vec",
        "save([1.0], \"https://example.com/x.npy\")\n");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(!out.status.success());
    assert!(stderr.contains("cannot save to a url"), "{}", stderr);
}

#[test]
fn url_without_format_fails_loud() {
    let out = run_vector_src("vector_url_noext.vec",
        "print(load(\"https://example.com/data\"))\n");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(!out.status.success());
    assert!(stderr.contains("must end in"), "{}", stderr);
}

#[test]
fn wav_load_decodes_reference_files() {
    let out = run_vector_src("vector_wav_load.vec",
        "a = load(\"tests/fixtures/mono16.wav\")\nprint(a.samples * 32768.0)\nprint(a.rate)\ns = load(\"tests/fixtures/stereo8.wav\")\nprint(s.samples * 128.0)\nt = load(\"tests/fixtures/mono24.wav\")\nprint(t.samples)\n");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        "[0, 16384, -16384, 32767] : f32\n\
         8000 : f32\n\
         [[0, 127], [-128, -64]] : f32\n\
         [0.5, -0.5] : f32\n"
    );
}

#[test]
fn wav_save_roundtrip() {
    fs::create_dir_all("tests/cases/data").unwrap();
    let a = run_vector_src("vector_wav_a.vec",
        "save({samples: [0.0, 0.5, -0.5], rate: 8000.0}, \"tests/cases/data/rt.wav\")\nprint(1.0)\n");
    assert!(a.status.success(), "{}", String::from_utf8_lossy(&a.stderr));
    let b = run_vector_src("vector_wav_b.vec",
        "c = load(\"tests/cases/data/rt.wav\")\nprint(c.samples)\nprint(c.rate)\n");
    assert!(b.status.success(), "{}", String::from_utf8_lossy(&b.stderr));
    assert_eq!(String::from_utf8(b.stdout).unwrap(), "[0, 0.5, -0.5] : f32\n8000 : f32\n");
}

#[test]
fn wav_save_requires_samples_and_rate() {
    let out = run_vector_src("vector_wav_bad.vec",
        "save({samples: [0.1, 0.2]}, \"tests/cases/data/bad.wav\")\n");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(!out.status.success());
    assert!(stderr.contains("{samples, rate}"), "{}", stderr);
}

#[test]
fn compressed_audio_fails_loud() {
    let out = run_vector_src("vector_mp3.vec", "print(load(\"song.mp3\"))\n");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(!out.status.success());
    assert!(stderr.contains("convert to wav"), "{}", stderr);
}

#[test]
fn jpeg_fails_loud() {
    let out = run_vector_src("vector_jpeg.vec", "print(load(\"photo.jpg\"))\n");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(!out.status.success());
    assert!(stderr.contains("convert to png"), "{}", stderr);
}

#[test]
fn plot_writes_svg() {
    fs::create_dir_all("tests/cases/data").unwrap();
    let out = run_vector_src("vector_plot.vec",
        "x = linspace(0.0, 1.0, 5)\nplot(x, x * x, \"quad\")\nscatter(x, x)\ntitle(\"curves\")\nxlabel(\"x\")\nylabel(\"y\")\nsavefig(\"tests/cases/data/fig.svg\")\nprint(sum(x))\n");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(String::from_utf8(out.stdout).unwrap(), "2.5 : f32\n");
    let svg = fs::read_to_string("tests/cases/data/fig.svg").unwrap();
    assert!(svg.starts_with("<svg"), "{}", svg);
    assert!(svg.contains("<polyline"), "{}", svg);
    assert!(svg.contains("<circle"), "{}", svg);
    assert!(svg.contains(">curves</text>"), "{}", svg);
    assert!(svg.contains(">quad</text>"), "{}", svg);
    assert!(svg.trim_end().ends_with("</svg>"), "{}", svg);
}

#[test]
fn plot_without_savefig_fails_loud() {
    let out = run_vector_src("vector_plot_unfinished.vec",
        "x = linspace(0.0, 1.0, 5)\nplot(x, x)\nprint(sum(x))\n");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(!out.status.success());
    assert!(stderr.contains("savefig"), "{}", stderr);
}

#[test]
fn savefig_without_plot_fails_loud() {
    let out = run_vector_src("vector_savefig_empty.vec",
        "savefig(\"tests/cases/data/empty.svg\")\nprint(1.0)\n");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(!out.status.success());
    assert!(stderr.contains("without any plot"), "{}", stderr);
}

#[test]
fn export_requires_module_instance() {
    let out = run_vector_src("vector_export_tensor.vec",
        "x = [1.0]\nexport(x, \"tests/cases/data/x.mlir\", [1.0])\n");
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert!(!out.status.success());
    assert!(stderr.contains("module instance"), "{}", stderr);
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
fn repl_imports_libraries() {
    fs::create_dir_all("tests/cases/data").unwrap();
    fs::write("tests/cases/data/repl_lib.vec", "fn triple(x):\n  x * 3.0\n").unwrap();
    let (stdout, stderr) = run_repl_script(
        "import tests.cases.data.repl_lib\ntriple(7.0)\n",
    );
    assert_eq!(stdout, "21 : f32\n", "stderr: {}", stderr);
}

#[test]
fn repl_redefines_functions() {
    let (stdout, _) = run_repl_script(
        "fn f(x):\n  x * 2.0\n\nsum(f(3.0))\nfn f(x):\n  x * 10.0\n\nsum(f(3.0))\n",
    );
    assert_eq!(stdout, "6 : f32\n30 : f32\n");
}
