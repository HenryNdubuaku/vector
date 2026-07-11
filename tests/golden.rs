use std::fs;
use std::process::Command;

fn run_vector(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_vector")).args(args).output().unwrap()
}

#[test]
fn golden_cases() {
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
fn matmul_contraction_mismatch_fails() {
    let path = std::env::temp_dir().join("vector_matmul_mismatch.vec");
    fs::write(&path, "print(matmul([[1.0, 2.0]], [[1.0, 2.0]]))\n").unwrap();
    let output = run_vector(&["run", path.to_str().unwrap()]);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(!output.status.success());
    assert!(stderr.contains("matmul"), "{}", stderr);
}
