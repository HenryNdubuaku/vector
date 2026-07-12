use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::die;
use crate::lexer::lex;
use crate::parser::{Decl, Expr, ModuleDecl, Parser, Program};

fn parse_library(path: &Path) -> Program {
    let src = fs::read_to_string(path)
        .unwrap_or_else(|e| die(&format!("cannot read import {}: {}", path.display(), e)));
    let lexed = lex(&src);
    let mut p = Parser {
        repl: false,
        library: true,
        toks: lexed.toks,
        cols: lexed.cols,
        lines: lexed.lines,
        pos: 0,
        imports: Vec::new(),
        fns: HashMap::new(),
        modules: HashMap::new(),
    };
    p.program()
}

pub fn load_libraries(from: &str, imports: &[String]) -> (HashMap<String, Decl>, HashMap<String, ModuleDecl>) {
    let mut fns = HashMap::new();
    let mut modules = HashMap::new();
    let mut visiting = Vec::new();
    let mut done = HashSet::new();
    let base = Path::new(from).parent().unwrap_or(Path::new("")).to_path_buf();
    for imp in imports {
        resolve(&base.join(imp), &mut fns, &mut modules, &mut visiting, &mut done);
    }
    (fns, modules)
}

fn resolve(
    path: &Path,
    fns: &mut HashMap<String, Decl>,
    modules: &mut HashMap<String, ModuleDecl>,
    visiting: &mut Vec<PathBuf>,
    done: &mut HashSet<PathBuf>,
) {
    let canon = fs::canonicalize(path)
        .unwrap_or_else(|e| die(&format!("cannot read import {}: {}", path.display(), e)));
    if done.contains(&canon) {
        return;
    }
    if visiting.contains(&canon) {
        die(&format!("circular import: {}", path.display()));
    }
    visiting.push(canon.clone());
    let prog = parse_library(path);
    if !matches!(prog.main, Expr::Unit) {
        die(&format!("{} has top-level code; imported files hold only fn and module declarations", path.display()));
    }
    let base = path.parent().unwrap_or(Path::new("")).to_path_buf();
    for imp in &prog.imports {
        resolve(&base.join(imp), fns, modules, visiting, done);
    }
    for (name, decl) in prog.fns {
        if fns.contains_key(&name) || modules.contains_key(&name) {
            die(&format!("{} is defined in more than one imported file", name));
        }
        fns.insert(name, decl);
    }
    for (name, decl) in prog.modules {
        if fns.contains_key(&name) || modules.contains_key(&name) {
            die(&format!("{} is defined in more than one imported file", name));
        }
        modules.insert(name, decl);
    }
    visiting.pop();
    done.insert(canon);
}
