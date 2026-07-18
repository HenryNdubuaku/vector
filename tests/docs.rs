use std::fs;

fn builtin_names() -> Vec<String> {
    let src = fs::read_to_string("src/builtins.rs").unwrap();
    let mut names = Vec::new();
    for line in src.lines() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with('"') || !trimmed.trim_end().ends_with("=> {") {
            continue;
        }
        for part in trimmed.split('|') {
            let part = part.trim();
            if let Some(name) = part.strip_prefix('"').and_then(|p| p.split('"').next()) {
                if name.chars().all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit()) {
                    names.push(name.to_string());
                }
            }
        }
    }
    names
}

fn stdlib_names() -> Vec<String> {
    let src = fs::read_to_string("src/linear.rs").unwrap();
    let mut names = Vec::new();
    for line in src.lines() {
        let trimmed = line.trim();
        if !trimmed.ends_with(':') {
            continue;
        }
        for prefix in ["function ", "module "] {
            if let Some(rest) = trimmed.strip_prefix(prefix) {
                if let Some(name) = rest.split('(').next() {
                    names.push(name.trim().to_string());
                }
            }
        }
    }
    names.retain(|n| !n.contains(' ') && !n.is_empty());
    names
}

#[test]
fn every_builtin_is_documented() {
    let reference = fs::read_to_string("docs/reference.md").unwrap();
    let mut missing = Vec::new();
    for name in builtin_names().into_iter().chain(stdlib_names()) {
        if !reference.contains(&format!("`{}(", name)) {
            missing.push(name);
        }
    }
    assert!(
        missing.is_empty(),
        "undocumented builtins (add them to docs/reference.md as `name(...)`): {}",
        missing.join(", ")
    );
}

#[test]
fn examples_doc_is_generated_from_golden_cases() {
    let mut cases: Vec<String> = fs::read_dir("tests/cases")
        .unwrap()
        .filter_map(|e| {
            let path = e.unwrap().path();
            if path.extension().and_then(|x| x.to_str()) == Some("vec") {
                Some(path.file_stem().unwrap().to_string_lossy().into_owned())
            } else {
                None
            }
        })
        .collect();
    cases.sort();
    let mut doc = String::from(
        "# Vector by example\n\n\
         Generated from the test suite by `cargo test` — every program and its output are verified on each run. Do not edit by hand.\n",
    );
    for case in &cases {
        let program = fs::read_to_string(format!("tests/cases/{}.vec", case)).unwrap();
        let output = fs::read_to_string(format!("tests/cases/{}.out", case)).unwrap();
        doc.push_str(&format!(
            "\n## {}\n\n```python\n{}```\n\nOutput:\n\n```\n{}```\n",
            case, program, output
        ));
    }
    fs::write("docs/examples.md", doc).unwrap();
    assert!(cases.len() >= 20, "golden cases went missing");
}
