use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead, IsTerminal, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::emit::build_module;
use crate::graph::{Dtype, InputSource, ModTag, OpKind, TVal, Val};
use crate::lexer::{lex, Tok};
use crate::linear;
use crate::npy::{input_host_buffer, InputSpec};
use crate::parser::{Decl, Expr, ModuleDecl, Parser};
use crate::runtime::{Engine, Tensor};
use crate::safetensors::write_save;
use crate::trace::Tracer;

enum SessionVal {
    Const(f64, Dtype),
    Value(Tensor),
    Record(Option<ModTag>, Vec<(String, SessionVal)>),
}

struct Session {
    engine: Engine,
    fns: HashMap<String, Decl>,
    modules: HashMap<String, ModuleDecl>,
    env: HashMap<String, SessionVal>,
    rng: u64,
}

pub fn run_repl() {
    let mut session = Session {
        engine: Engine::new(),
        fns: HashMap::new(),
        modules: linear::stdlib_modules(),
        env: HashMap::new(),
        rng: 0x243F6A8885A308D3,
    };
    let stdin = io::stdin();
    let tty = stdin.is_terminal();
    if tty {
        println!("vector {} — exit with 'exit' or ctrl-d", env!("CARGO_PKG_VERSION"));
    }
    let mut lines = stdin.lock().lines();
    loop {
        if tty {
            print!(">>> ");
            let _ = io::stdout().flush();
        }
        let Some(Ok(first)) = lines.next() else { break };
        if first.trim().is_empty() {
            continue;
        }
        if first.trim() == "exit" || first.trim() == "quit" {
            break;
        }
        let mut chunk = first;
        let block = chunk.trim_end().ends_with(':');
        if block || unbalanced(&chunk) {
            loop {
                if tty {
                    print!("... ");
                    let _ = io::stdout().flush();
                }
                let Some(Ok(line)) = lines.next() else { break };
                if line.trim().is_empty() && !unbalanced(&chunk) {
                    break;
                }
                chunk.push('\n');
                chunk.push_str(&line);
                if !block && !unbalanced(&chunk) {
                    break;
                }
            }
        }
        let _ = catch_unwind(AssertUnwindSafe(|| eval_chunk(&mut session, &chunk)));
    }
}

fn unbalanced(chunk: &str) -> bool {
    let mut depth = 0i64;
    for line in chunk.lines() {
        let mut in_string = false;
        for c in line.chars() {
            match c {
                '"' => in_string = !in_string,
                '#' if !in_string => break,
                '(' | '[' | '{' if !in_string => depth += 1,
                ')' | ']' | '}' if !in_string => depth -= 1,
                _ => {}
            }
        }
    }
    depth > 0
}

fn eval_chunk(session: &mut Session, chunk: &str) {
    let lexed = lex(chunk);
    let used: HashSet<String> = lexed.toks.iter()
        .filter_map(|t| match t {
            Tok::Ident(s) => Some(s.clone()),
            _ => None,
        })
        .collect();
    let mut p = Parser {
        repl: true,
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
    let (import_fns, import_modules) = crate::imports::load_libraries("", &prog.imports);
    session.fns.extend(import_fns);
    session.modules.extend(import_modules);
    session.fns.extend(prog.fns);
    session.modules.extend(prog.modules);

    let mut tracer = Tracer {
        nodes: Vec::new(),
        prints: Vec::new(),
        inputs: Vec::new(),
        saves: Vec::new(),
        exports: Vec::new(),
        figures: Vec::new(),
        figure: crate::plot::FigureSpec::default(),
        plays: Vec::new(),
        loop_prints: Vec::new(),
        modules: session.modules.clone(),
        statics: Vec::new(),
        rng: session.rng,
        claimed: HashSet::new(),
        region_depth: 0,
        grad_depth: 0,
        interned: vec![HashMap::new()],
    };

    let mut feed_map: HashMap<String, Tensor> = HashMap::new();
    let mut env: HashMap<String, TVal> = HashMap::new();
    for (name, sv) in &session.env {
        if used.contains(name) {
            let tv = inject(&mut tracer, sv, name, &mut feed_map);
            env.insert(name.clone(), tv);
        }
    }

    let mut bound: Vec<String> = Vec::new();
    let mut echo: Option<TVal> = None;
    walk(&mut tracer, &prog.main, &mut env, &session.fns, &mut bound, &mut echo);
    if !tracer.figure.series.is_empty() {
        crate::die("plot without savefig or show; finish the figure");
    }

    let mut metas: Vec<Meta> = Vec::new();
    let mut outputs: Vec<Val> = Vec::new();
    for spec in tracer.prints.clone() {
        metas.push(Meta::Show(spec.label, spec.rows));
        outputs.push(spec.val);
    }
    if let Some(v) = &echo {
        flatten_show(v, None, &mut metas, &mut outputs);
    }
    for name in &bound {
        let tv = env[name].clone();
        plan_binding(&tracer, &tv, &mut metas, &mut outputs);
    }
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

    let mut results: Vec<Tensor> = Vec::new();
    if !outputs.is_empty() {
        let params: Vec<usize> = tracer.inputs.iter().map(|&(_, id)| id).collect();
        let module = build_module(&tracer.nodes, &params, &outputs);
        let feeds = tracer.inputs.iter()
            .map(|(src, id)| match src {
                InputSource::Npy(path) | InputSource::Image(path) | InputSource::Audio(path) => input_host_buffer(&InputSpec {
                    path: path.clone(),
                    entry: None,
                    shape: tracer.nodes[*id].shape.clone(),
                    dtype: tracer.nodes[*id].dtype,
                }),
                InputSource::Safetensors(path, name) | InputSource::Csv(path, name) => input_host_buffer(&InputSpec {
                    path: path.clone(),
                    entry: Some(name.clone()),
                    shape: tracer.nodes[*id].shape.clone(),
                    dtype: tracer.nodes[*id].dtype,
                }),
                InputSource::Live(key) => feed_map[key].to_host_buffer(),
            })
            .collect();
        results = session.engine.execute(&module, feeds);
    }

    let mut results = results.into_iter();
    let mut captured: Vec<Tensor> = Vec::new();
    for meta in &metas {
        match meta {
            Meta::Show(label, rows) => {
                let t = results.next().unwrap();
                crate::print_result(label, rows, &t);
            }
            Meta::Capture => captured.push(results.next().unwrap()),
        }
    }

    for spec in &tracer.saves {
        let tensors: Vec<Tensor> = spec.names.iter().map(|_| results.next().unwrap()).collect();
        write_save(spec, &tensors);
    }
    for spec in &tracer.exports {
        let tensors: Vec<Tensor> = spec.weight_vals.iter().map(|_| results.next().unwrap()).collect();
        crate::export::write_export(spec, &tensors);
    }
    for (i, fig) in tracer.figures.iter().enumerate() {
        let count = fig.series.len() * 2 + fig.images.len();
        let tensors: Vec<Tensor> = (0..count).map(|_| results.next().unwrap()).collect();
        let written = crate::plot::write_figure(fig, &tensors, i);
        if fig.path.is_none() {
            crate::open_figure(&written);
        }
    }
    for spec in &tracer.plays {
        let tensors: Vec<Tensor> = spec.vals.iter().map(|_| results.next().unwrap()).collect();
        crate::audio::write_wav(spec, &tensors);
        crate::play_audio(&spec.path);
    }

    let mut captured = captured.into_iter();
    for name in &bound {
        let tv = env[name].clone();
        let sv = persist(&tracer, &tv, &mut captured);
        session.env.insert(name.clone(), sv);
    }
    session.rng = tracer.rng;
}

enum Meta {
    Show(Option<String>, Option<crate::trace::RowMeta>),
    Capture,
}

fn inject(tracer: &mut Tracer, sv: &SessionVal, key: &str, feeds: &mut HashMap<String, Tensor>) -> TVal {
    match sv {
        SessionVal::Const(v, dtype) => {
            let val = tracer.constant(*v, *dtype);
            TVal::Tensor(crate::graph::BVal { val, bdims: 0 })
        }
        SessionVal::Value(t) => {
            let val = tracer.live_input(key.to_string(), t.shape().to_vec(), t.graph_dtype());
            feeds.insert(key.to_string(), t.clone());
            TVal::Tensor(crate::graph::BVal { val, bdims: 0 })
        }
        SessionVal::Record(tag, fields) => {
            let mut out = Vec::new();
            for (k, f) in fields {
                let child = inject(tracer, f, &format!("{}.{}", key, k), feeds);
                out.push((k.clone(), child));
            }
            TVal::Record(tag.clone(), out)
        }
    }
}

fn walk(
    tracer: &mut Tracer,
    e: &Expr,
    env: &mut HashMap<String, TVal>,
    fns: &HashMap<String, Decl>,
    bound: &mut Vec<String>,
    echo: &mut Option<TVal>,
) {
    match e {
        Expr::Unit => {}
        Expr::Let(name, value, rest) => {
            let v = tracer.trace(value, env, fns);
            env.insert(name.clone(), v);
            if !bound.contains(name) {
                bound.push(name.clone());
            }
            walk(tracer, rest, env, fns, bound, echo);
        }
        Expr::Seq(first, rest) => {
            tracer.trace(first, env, fns);
            walk(tracer, rest, env, fns, bound, echo);
        }
        Expr::For(var, start, end, step, stmts, rest) => {
            let env3 = tracer.trace_for(var, start, end, step, stmts, env, fns);
            for (name, _) in stmts {
                if let Some(n) = name {
                    if env.contains_key(n) && !bound.contains(n) {
                        bound.push(n.clone());
                    }
                }
            }
            *env = env3;
            walk(tracer, rest, env, fns, bound, echo);
        }
        expr => {
            let v = tracer.trace(expr, env, fns);
            *echo = Some(v);
        }
    }
}

fn flatten_show(v: &TVal, label: Option<String>, metas: &mut Vec<Meta>, outputs: &mut Vec<Val>) {
    match v {
        TVal::Tensor(b) => {
            metas.push(Meta::Show(label, None));
            outputs.push(b.val.clone());
        }
        TVal::Record(_, fields) => {
            for (k, f) in fields {
                let path = match &label {
                    Some(p) => format!("{}.{}", p, k),
                    None => k.clone(),
                };
                flatten_show(f, Some(path), metas, outputs);
            }
        }
    }
}

fn is_const_scalar(tracer: &Tracer, val: &Val) -> Option<f64> {
    if !val.shape.is_empty() {
        return None;
    }
    match tracer.nodes[val.id].kind {
        OpKind::Constant(n) => Some(n),
        _ => None,
    }
}

fn plan_binding(tracer: &Tracer, tv: &TVal, metas: &mut Vec<Meta>, outputs: &mut Vec<Val>) {
    match tv {
        TVal::Tensor(b) => {
            if is_const_scalar(tracer, &b.val).is_none() {
                metas.push(Meta::Capture);
                outputs.push(b.val.clone());
            }
        }
        TVal::Record(_, fields) => {
            for (_, f) in fields {
                plan_binding(tracer, f, metas, outputs);
            }
        }
    }
}

fn persist(tracer: &Tracer, tv: &TVal, captured: &mut std::vec::IntoIter<Tensor>) -> SessionVal {
    match tv {
        TVal::Tensor(b) => match is_const_scalar(tracer, &b.val) {
            Some(n) => SessionVal::Const(n, b.val.dtype),
            None => SessionVal::Value(captured.next().unwrap()),
        },
        TVal::Record(tag, fields) => {
            let mut out = Vec::new();
            for (k, f) in fields {
                out.push((k.clone(), persist(tracer, f, captured)));
            }
            SessionVal::Record(tag.clone(), out)
        }
    }
}
