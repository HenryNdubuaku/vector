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
        let output = run_vector(&["run", path.to_str().unwrap()]);
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
    let output = run_vector(&["run", path.to_str().unwrap()]);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success());
    assert!(stderr.contains("[2]") && stderr.contains("[3]"), "{}", stderr);
}

#[test]
fn size_one_stretching_is_rejected() {
    let path = std::env::temp_dir().join("vector_stretch.vec");
    fs::write(&path, "print([[1.0], [2.0]] * [[1.0, 2.0], [3.0, 4.0]])\n").unwrap();
    let output = run_vector(&["run", path.to_str().unwrap()]);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success());
    assert!(stderr.contains("[2, 1]") && stderr.contains("[2, 2]"), "{}", stderr);
}

#[test]
fn grad_requires_scalar_output() {
    let path = std::env::temp_dir().join("vector_grad_nonscalar.vec");
    fs::write(&path, "fn f(x):\n  x * 2.0\n\nprint(grad(f, [1.0, 2.0]))\n").unwrap();
    let output = run_vector(&["run", path.to_str().unwrap()]);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success());
    assert!(stderr.contains("scalar") && stderr.contains("[2]"), "{}", stderr);
}

#[test]
fn load_missing_file_fails_loud() {
    let path = std::env::temp_dir().join("vector_load_missing.vec");
    fs::write(&path, "print(load(\"/nonexistent/data.npy\"))\n").unwrap();
    let output = run_vector(&["run", path.to_str().unwrap()]);
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
    let output = run_vector(&["run", path.to_str().unwrap()]);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success());
    assert!(stderr.contains("batching depth"), "{}", stderr);
}

#[test]
fn matmul_contraction_mismatch_fails() {
    let path = std::env::temp_dir().join("vector_matmul_mismatch.vec");
    fs::write(&path, "print(matmul([[1.0, 2.0]], [[1.0, 2.0]]))\n").unwrap();
    let output = run_vector(&["run", path.to_str().unwrap()]);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success());
    assert!(stderr.contains("matmul"), "{}", stderr);
}
