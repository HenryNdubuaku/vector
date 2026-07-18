use std::collections::HashMap;

use crate::batch::{collect_leaves, rebuild};
use crate::die;
use crate::graph::{broadcast_shape, per_shape, BVal, Dtype, InputSource, ModTag, OpKind, TVal, Val};
use crate::npy::npy_meta;
use crate::parser::{Decl, Expr};
use crate::trace::Tracer;

impl Tracer {
    pub fn instantiate(&mut self, name: &str, args: &[Expr], env: &HashMap<String, TVal>, fns: &HashMap<String, Decl>) -> TVal {
        let decl = self.modules.get(name).unwrap().clone();
        if args.len() != decl.params.len() {
            die(&format!("module {} expects {} args, got {}", name, decl.params.len(), args.len()));
        }
        let statics: Vec<(String, f64)> = decl.params.iter()
            .zip(args)
            .map(|(p, a)| (p.clone(), self.num_lit(a, env, &format!("module {} argument", name))))
            .collect();
        self.statics.push(statics.iter().cloned().collect());
        let mut env2: HashMap<String, TVal> = HashMap::new();
        let mut fields: Vec<(String, TVal)> = Vec::new();
        for (fname, expr) in &decl.init {
            let v = self.trace(expr, &env2, fns);
            env2.insert(fname.clone(), v.clone());
            match fields.iter_mut().find(|(k, _)| k == fname) {
                Some(field) => field.1 = v,
                None => fields.push((fname.clone(), v)),
            }
        }
        self.statics.pop();
        TVal::Record(Some(ModTag { module: name.to_string(), statics }), fields)
    }

    pub fn call_method(&mut self, callee: TVal, method: &str, args: Vec<TVal>, fns: &HashMap<String, Decl>) -> TVal {
        let tag = match &callee {
            TVal::Record(Some(t), _) => t.clone(),
            _ => die("value is not callable (only module instances can be applied)"),
        };
        let module = self.modules.get(&tag.module)
            .unwrap_or_else(|| die(&format!("unknown module: {}", tag.module)))
            .clone();
        let decl = module.method(method)
            .unwrap_or_else(|| die(&format!("module {} has no method {}", tag.module, method)));
        if args.len() != decl.params.len() - 1 {
            die(&format!("{}.{} expects {} args, got {}",
                         tag.module, method, decl.params.len() - 1, args.len()));
        }
        let mut env2 = HashMap::new();
        env2.insert(decl.params[0].clone(), callee);
        for (p, v) in decl.params[1..].iter().zip(args) {
            env2.insert(p.clone(), v);
        }
        self.statics.push(tag.statics.iter().cloned().collect());
        let out = self.trace(&decl.body, &env2, fns);
        self.statics.pop();
        out
    }

    pub fn builtin(&mut self, name: &str, args: &[Expr], env: &HashMap<String, TVal>, fns: &HashMap<String, Decl>) -> TVal {
        match name {
            "sum" | "mean" | "max" | "min" => {
                if args.is_empty() || args.len() > 2 {
                    die(&format!("{} expects 1 or 2 args, got {}", name, args.len()));
                }
                let v = self.trace(&args[0], env, fns).tensor(name);
                let per = per_shape(&v);
                let per_axes: Vec<usize> = if args.len() == 2 {
                    vec![self.axis_lit(&args[1], env, &per)]
                } else {
                    (0..per.len()).collect()
                };
                let axes: Vec<usize> = per_axes.iter().map(|a| a + v.bdims).collect();
                let (reducer, init) = match name {
                    "max" => ("maximum", f64::NEG_INFINITY),
                    "min" => ("minimum", f64::INFINITY),
                    _ => ("add", 0.0),
                };
                let total = self.reduce(reducer, init, &v.val, &axes);
                let total = TVal::Tensor(BVal { val: total, bdims: v.bdims });
                if name != "mean" {
                    return total;
                }
                let count: usize = per_axes.iter().map(|&d| per[d]).product();
                let denom = self.constant(count as f64, v.val.dtype);
                self.tmap2("divide", total, TVal::Tensor(BVal { val: denom, bdims: 0 }))
            }
            "tokenize" => {
                if args.len() != 2 {
                    die(&format!("tokenize expects (\"data.txt\", \"tokenizer.json\"), got {} args", args.len()));
                }
                let (txt, tok) = match (&args[0], &args[1]) {
                    (Expr::Str(a), Expr::Str(b)) => (a.clone(), b.clone()),
                    _ => die("tokenize expects two file path string literals"),
                };
                let txt = if crate::net::is_url(&txt) { crate::net::fetch(&txt) } else { txt };
                let tok = if crate::net::is_url(&tok) { crate::net::fetch(&tok) } else { tok };
                if let Some(&(_, id)) = self.inputs.iter()
                    .find(|(src, _)| matches!(src, InputSource::Tokens(t, k) if *t == txt && *k == tok)) {
                    return TVal::Tensor(BVal { val: self.val(id), bdims: 0 });
                }
                let n = crate::text::encode_file(&txt, &tok).len();
                let val = self.emit(OpKind::Input, vec![], vec![n], Dtype::F32);
                self.inputs.push((InputSource::Tokens(txt, tok), val.id));
                TVal::Tensor(BVal { val, bdims: 0 })
            }
            "detokenize" => {
                if args.len() != 2 {
                    die(&format!("detokenize expects (ids, \"tokenizer.json\"), got {} args", args.len()));
                }
                let tok = match &args[1] {
                    Expr::Str(s) => s.clone(),
                    _ => die("detokenize expects a tokenizer path string literal"),
                };
                let tok = if crate::net::is_url(&tok) { crate::net::fetch(&tok) } else { tok };
                crate::text::check_tokenizer(&tok);
                let v = self.trace(&args[0], env, fns);
                let b = v.tensor("detokenize");
                self.decodes.insert(b.val.id, crate::trace::Decode::Tokens(tok));
                TVal::Tensor(b)
            }
            "text" => {
                if args.len() != 1 {
                    die(&format!("text expects 1 arg, got {}", args.len()));
                }
                let v = self.trace(&args[0], env, fns);
                let b = v.tensor("text");
                self.decodes.insert(b.val.id, crate::trace::Decode::Bytes);
                TVal::Tensor(b)
            }
            "bincount" => {
                if args.len() != 2 {
                    die(&format!("bincount expects (values, bins), got {} args", args.len()));
                }
                let v = self.trace(&args[0], env, fns).tensor("bincount");
                if v.bdims != 0 {
                    die("bincount inside vmap isn't supported yet");
                }
                if v.val.shape.len() != 1 {
                    die("bincount expects a vector");
                }
                let bins = self.int_lit(&args[1], env, "bincount bins");
                if bins == 0 {
                    die("bincount needs at least one bin");
                }
                let n = v.val.shape[0];
                let idx = self.convert(&v.val, Dtype::I64);
                let zeros = self.zeros(&[bins], Dtype::F32);
                let one = self.constant(1.0, Dtype::F32);
                let ones = self.broadcast(&one, &[n]);
                let val = self.emit(OpKind::Scatter, vec![zeros.id, idx.id, ones.id], vec![bins], Dtype::F32);
                TVal::Tensor(BVal { val, bdims: 0 })
            }
            "print" => {
                if args.len() != 1 {
                    die(&format!("print expects 1 arg, got {}", args.len()));
                }
                let v = self.trace(&args[0], env, fns);
                if let TVal::Tensor(b) = &v {
                    if b.val.dtype == Dtype::I1 {
                        die("cannot print booleans; use where to select values");
                    }
                }
                if self.region_depth > 0 {
                    self.push_loop_prints(None, &v);
                } else {
                    self.push_prints(None, &v);
                }
                v
            }
            "where" => {
                if args.len() != 3 {
                    die(&format!("where expects 3 args (condition, then, else), got {}", args.len()));
                }
                let c = self.trace(&args[0], env, fns).tensor("where condition");
                if c.val.dtype != Dtype::I1 {
                    die("where condition must be a comparison");
                }
                let a = self.trace(&args[1], env, fns).tensor("where branch");
                let b = self.trace(&args[2], env, fns).tensor("where branch");
                if a.val.dtype == Dtype::I1 || b.val.dtype == Dtype::I1 {
                    die("where branches cannot be booleans");
                }
                let per = broadcast_shape(&broadcast_shape(&per_shape(&a), &per_shape(&b)), &per_shape(&c));
                let deepest = [&a, &b, &c].into_iter().max_by_key(|v| v.bdims).unwrap();
                let prefix: Vec<usize> = deepest.val.shape[..deepest.bdims].to_vec();
                let cv = self.align(&c, &prefix, &per);
                let (a, b) = if a.val.dtype == b.val.dtype {
                    (a, b)
                } else if per_shape(&a).is_empty() {
                    let av = self.convert(&a.val, b.val.dtype);
                    (BVal { val: av, bdims: a.bdims }, b)
                } else if per_shape(&b).is_empty() {
                    let bv = self.convert(&b.val, a.val.dtype);
                    (a, BVal { val: bv, bdims: b.bdims })
                } else {
                    die(&format!("dtype mismatch: {} vs {}", a.val.dtype.name(), b.val.dtype.name()));
                };
                let av = self.align(&a, &prefix, &per);
                let bv = self.align(&b, &prefix, &per);
                let mut shape = prefix.clone();
                shape.extend(&per);
                let dtype = av.dtype;
                let val = self.emit(OpKind::Select, vec![cv.id, av.id, bv.id], shape, dtype);
                TVal::Tensor(BVal { val, bdims: prefix.len() })
            }
            "f32" | "f64" => {
                if args.len() != 1 {
                    die(&format!("{} expects 1 arg, got {}", name, args.len()));
                }
                let dtype = if name == "f32" { Dtype::F32 } else { Dtype::F64 };
                let v = self.trace(&args[0], env, fns);
                self.tconvert(&v, dtype)
            }
            "slice" => {
                if args.len() != 3 {
                    die(&format!("slice expects (x, start, size), got {} args", args.len()));
                }
                let x = self.trace(&args[0], env, fns).tensor("slice");
                if x.bdims != 0 {
                    die("slice inside vmap isn't supported yet");
                }
                if x.val.shape.is_empty() {
                    die("slice needs rank >= 1");
                }
                let start = self.trace(&args[1], env, fns).tensor("slice start");
                if start.bdims != 0 || start.val.shape.len() > 1 {
                    die("slice start must be a scalar or a vector of starts");
                }
                let size = self.int_lit(&args[2], env, "slice size");
                if size == 0 || size > x.val.shape[0] {
                    die(&format!("slice size {} out of range for leading dim {}", size, x.val.shape[0]));
                }
                if let [count] = start.val.shape[..] {
                    let idx = self.convert(&start.val, Dtype::I64);
                    let mut shape = vec![count, size];
                    shape.extend(&x.val.shape[1..]);
                    let val = self.emit(OpKind::Gather(size), vec![x.val.id, idx.id], shape.clone(), x.val.dtype);
                    return TVal::Tensor(BVal { val, bdims: 0 });
                }
                let idx = self.convert(&start.val, Dtype::I64);
                let iota = self.emit(OpKind::Iota, vec![], vec![size], Dtype::I32);
                let offsets = self.convert(&iota, Dtype::I64);
                let indices = self.ewise("add", offsets, idx);
                let mut shape = vec![size];
                shape.extend(&x.val.shape[1..]);
                let val = self.emit(OpKind::Gather(1), vec![x.val.id, indices.id], shape.clone(), x.val.dtype);
                TVal::Tensor(BVal { val, bdims: 0 })
            }
            "exp" | "log" | "tanh" | "sqrt" | "sin" | "cos" | "floor" => {
                if args.len() != 1 {
                    die(&format!("{} expects 1 arg, got {}", name, args.len()));
                }
                let v = self.trace(&args[0], env, fns);
                let op = match name {
                    "exp" => "exponential",
                    "sin" => "sine",
                    "cos" => "cosine",
                    _ => name,
                };
                self.tunary(op, &v)
            }
            "arange" => {
                if args.is_empty() || args.len() > 3 {
                    die(&format!("arange expects (stop), (start, stop) or (start, stop, step), got {} args", args.len()));
                }
                let lits: Vec<f64> = args.iter().map(|a| self.num_lit(a, env, "arange argument")).collect();
                let (start, stop, step) = match lits.len() {
                    1 => (0.0, lits[0], 1.0),
                    2 => (lits[0], lits[1], 1.0),
                    _ => (lits[0], lits[1], lits[2]),
                };
                if step == 0.0 {
                    die("arange step must be nonzero");
                }
                let count = ((stop - start) / step).ceil();
                if count <= 0.0 {
                    die(&format!("arange({}, {}, {}) is empty", start, stop, step));
                }
                let indices = self.emit(OpKind::Iota, vec![], vec![count as usize], Dtype::F32);
                let step_c = self.constant(step, Dtype::F32);
                let scaled = self.ewise("multiply", indices, step_c);
                let start_c = self.constant(start, Dtype::F32);
                let val = self.ewise("add", scaled, start_c);
                TVal::Tensor(BVal { val, bdims: 0 })
            }
            "linspace" => {
                if args.len() != 3 {
                    die(&format!("linspace expects (start, stop, count), got {} args", args.len()));
                }
                let start = self.num_lit(&args[0], env, "linspace start");
                let stop = self.num_lit(&args[1], env, "linspace stop");
                let count = self.int_lit(&args[2], env, "linspace count");
                if count == 0 {
                    die("linspace count must be at least 1");
                }
                let step = if count == 1 { 0.0 } else { (stop - start) / (count - 1) as f64 };
                let indices = self.emit(OpKind::Iota, vec![], vec![count], Dtype::F32);
                let step_c = self.constant(step, Dtype::F32);
                let scaled = self.ewise("multiply", indices, step_c);
                let start_c = self.constant(start, Dtype::F32);
                let val = self.ewise("add", scaled, start_c);
                TVal::Tensor(BVal { val, bdims: 0 })
            }
            "reshape" => {
                if args.len() < 2 {
                    die("reshape expects a value and dimension literals");
                }
                let v = self.trace(&args[0], env, fns).tensor("reshape");
                let dims: Vec<usize> = args[1..].iter().map(|a| self.int_lit(a, env, "reshape dimension")).collect();
                let per = per_shape(&v);
                if dims.iter().product::<usize>() != per.iter().product::<usize>() {
                    die(&format!("reshape from {:?} to {:?} changes element count", per, dims));
                }
                let mut shape: Vec<usize> = v.val.shape[..v.bdims].to_vec();
                shape.extend(&dims);
                let val = self.reshape(&v.val, shape);
                TVal::Tensor(BVal { val, bdims: v.bdims })
            }
            "glorot_uniform" | "glorot_normal" | "he_uniform" | "he_normal" | "lecun_uniform" | "lecun_normal" => {
                if args.len() != 2 {
                    die(&format!("{} expects (fan_in, fan_out), got {} args", name, args.len()));
                }
                let fan_in = self.int_lit(&args[0], env, "fan_in");
                let fan_out = self.int_lit(&args[1], env, "fan_out");
                self.initializer(name, fan_in, fan_out)
            }
            "zeros" | "randn" => {
                if args.is_empty() {
                    die(&format!("{} expects dimension literals", name));
                }
                let dims: Vec<usize> = args.iter().map(|a| self.int_lit(a, env, "dimension")).collect();
                if name == "zeros" {
                    let val = self.zeros(&dims, Dtype::F32);
                    return TVal::Tensor(BVal { val, bdims: 0 });
                }
                let count: usize = dims.iter().product();
                let vals: Vec<f64> = (0..count).map(|_| self.next_normal()).collect();
                let val = self.emit(OpKind::DenseConst(vals), vec![], dims, Dtype::F32);
                TVal::Tensor(BVal { val, bdims: 0 })
            }
            "mod" => {
                if args.len() != 2 {
                    die(&format!("mod expects 2 args, got {}", args.len()));
                }
                let a = self.trace(&args[0], env, fns);
                let b = self.trace(&args[1], env, fns);
                let quotient = self.tmap2("divide", a.clone(), b.clone());
                let floored = self.tunary("floor", &quotient);
                let whole = self.tmap2("multiply", floored, b);
                self.tmap2("subtract", a, whole)
            }
            "maximum" | "minimum" => {
                if args.len() != 2 {
                    die(&format!("{} expects 2 args, got {}", name, args.len()));
                }
                let a = self.trace(&args[0], env, fns);
                let b = self.trace(&args[1], env, fns);
                self.tmap2(name, a, b)
            }
            "abs" => {
                if args.len() != 1 {
                    die(&format!("abs expects 1 arg, got {}", args.len()));
                }
                let v = self.trace(&args[0], env, fns);
                self.tunary("abs", &v)
            }
            "len" => {
                if args.len() != 1 {
                    die(&format!("len expects 1 arg, got {}", args.len()));
                }
                let v = self.trace(&args[0], env, fns).tensor("len");
                let per = per_shape(&v);
                if per.is_empty() {
                    die("len needs rank >= 1");
                }
                let val = self.constant(per[0] as f64, Dtype::F32);
                TVal::Tensor(BVal { val, bdims: 0 })
            }
            "pow" => {
                if args.len() != 2 {
                    die(&format!("pow expects (base, exponent), got {} args", args.len()));
                }
                let a = self.trace(&args[0], env, fns);
                let b = self.trace(&args[1], env, fns);
                self.tmap2("power", a, b)
            }
            "concat" | "stack" => {
                if args.len() < 2 {
                    die(&format!("{} expects at least 2 args, got {}", name, args.len()));
                }
                let parts: Vec<BVal> = args.iter().map(|a| self.trace(a, env, fns).tensor(name)).collect();
                let dtype = parts[0].val.dtype;
                if parts.iter().any(|p| p.val.dtype != dtype) {
                    die(&format!("{} expects matching dtypes", name));
                }
                let k = parts.iter().map(|p| p.bdims).max().unwrap();
                let prefix: Vec<usize> = parts.iter()
                    .find(|p| p.bdims == k)
                    .map(|p| p.val.shape[..k].to_vec())
                    .unwrap();
                let vals: Vec<Val> = if name == "stack" {
                    if parts.iter().any(|p| per_shape(p) != per_shape(&parts[0])) {
                        die("stack expects matching shapes");
                    }
                    parts.iter().map(|p| {
                        let per = per_shape(p);
                        let aligned = self.align(p, &prefix, &per);
                        let mut shape = prefix.clone();
                        shape.push(1);
                        shape.extend(&per);
                        self.reshape(&aligned, shape)
                    }).collect()
                } else {
                    if parts.iter().any(|p| per_shape(p).is_empty()) {
                        die("concat expects rank >= 1; stack scalars instead");
                    }
                    if parts.iter().any(|p| per_shape(p)[1..] != per_shape(&parts[0])[1..]) {
                        die("concat expects matching trailing dimensions");
                    }
                    parts.iter().map(|p| {
                        let per = per_shape(p);
                        self.align(p, &prefix, &per)
                    }).collect()
                };
                let mut shape = vals[0].shape.clone();
                shape[k] = vals.iter().map(|v| v.shape[k]).sum();
                let inputs: Vec<usize> = vals.iter().map(|v| v.id).collect();
                let val = self.emit(OpKind::Concat(k), inputs, shape, dtype);
                TVal::Tensor(BVal { val, bdims: k })
            }
            "save" => {
                if args.len() != 2 {
                    die(&format!("save expects (value, \"path\"), got {} args", args.len()));
                }
                let path = match &args[1] {
                    Expr::Str(s) => s.clone(),
                    _ => die("save expects a file path string literal"),
                };
                let v = self.trace(&args[0], env, fns);
                self.plan_save(&v, &path);
                v
            }
            "plot" | "scatter" => {
                let mut exprs: Vec<&Expr> = args.iter().collect();
                let label = match exprs.last() {
                    Some(Expr::Str(s)) => {
                        let s = s.clone();
                        exprs.pop();
                        Some(s)
                    }
                    _ => None,
                };
                if exprs.is_empty() || exprs.len() > 2 {
                    die(&format!("{} expects (y), (x, y) or (x, y, \"label\")", name));
                }
                let data: Vec<TVal> = exprs.iter().map(|a| self.trace(a, env, fns)).collect();
                self.plot_series(name == "scatter", data, label)
            }
            "title" | "xlabel" | "ylabel" => {
                if args.len() != 1 {
                    die(&format!("{} expects 1 arg, got {}", name, args.len()));
                }
                let s = match &args[0] {
                    Expr::Str(s) => s.clone(),
                    _ => die(&format!("{} expects a string literal", name)),
                };
                self.figure_text(name, s);
                let val = self.constant(0.0, Dtype::F32);
                TVal::Tensor(BVal { val, bdims: 0 })
            }
            "savefig" => {
                if args.len() != 1 {
                    die(&format!("savefig expects 1 arg, got {}", args.len()));
                }
                let path = match &args[0] {
                    Expr::Str(s) => s.clone(),
                    _ => die("savefig expects a file path string literal"),
                };
                self.finish_figure(Some(path));
                let val = self.constant(0.0, Dtype::F32);
                TVal::Tensor(BVal { val, bdims: 0 })
            }
            "show" => {
                if !args.is_empty() {
                    die(&format!("show expects no args, got {}", args.len()));
                }
                self.finish_figure(None);
                let val = self.constant(0.0, Dtype::F32);
                TVal::Tensor(BVal { val, bdims: 0 })
            }
            "export" => {
                if args.len() < 2 {
                    die("export expects (model, \"path\", example inputs...)");
                }
                let path = match &args[1] {
                    Expr::Str(s) => s.clone(),
                    _ => die("export expects a file path string literal"),
                };
                let v = self.trace(&args[0], env, fns);
                let examples: Vec<TVal> = args[2..].iter().map(|a| self.trace(a, env, fns)).collect();
                self.plan_export(&v, &path, examples, fns);
                v
            }
            "load" => {
                if args.len() != 1 {
                    die(&format!("load expects 1 arg, got {}", args.len()));
                }
                let path = match &args[0] {
                    Expr::Str(s) => s.clone(),
                    _ => die("load expects a file path string literal"),
                };
                let path = if crate::net::is_url(&path) { crate::net::fetch(&path) } else { path };
                if let Some(spec) = self.saves.iter().find(|s| s.path == path) {
                    return spec.value.clone();
                }
                if path.ends_with(".safetensors") {
                    return self.load_safetensors(&path);
                }
                if path.ends_with(".csv") {
                    return self.load_csv(&path);
                }
                if path.ends_with(".png") {
                    return self.load_png(&path);
                }
                if path.ends_with(".jpg") || path.ends_with(".jpeg") {
                    die("jpeg isn't supported; convert to png");
                }
                if path.ends_with(".wav") {
                    return self.load_wav(&path);
                }
                if path.ends_with(".mp3") || path.ends_with(".flac") || path.ends_with(".ogg") {
                    die("compressed audio isn't supported; convert to wav (pcm)");
                }
                if path.ends_with(".txt") {
                    if let Some(&(_, id)) = self.inputs.iter()
                        .find(|(src, _)| matches!(src, InputSource::Text(p) if *p == path)) {
                        return TVal::Tensor(BVal { val: self.val(id), bdims: 0 });
                    }
                    let n = crate::text::txt_len(&path);
                    let val = self.emit(OpKind::Input, vec![], vec![n], Dtype::F32);
                    self.inputs.push((InputSource::Text(path), val.id));
                    return TVal::Tensor(BVal { val, bdims: 0 });
                }
                if path.ends_with(".gz") {
                    if let Some(&(_, id)) = self.inputs.iter()
                        .find(|(src, _)| matches!(src, InputSource::Npy(p) if *p == path)) {
                        return TVal::Tensor(BVal { val: self.val(id), bdims: 0 });
                    }
                    let shape = crate::npy::idx_meta(&path);
                    let val = self.emit(OpKind::Input, vec![], shape, Dtype::F32);
                    self.inputs.push((InputSource::Npy(path), val.id));
                    return TVal::Tensor(BVal { val, bdims: 0 });
                }
                if let Some(&(_, id)) = self.inputs.iter()
                    .find(|(src, _)| matches!(src, InputSource::Npy(p) if *p == path)) {
                    return TVal::Tensor(BVal { val: self.val(id), bdims: 0 });
                }
                let (shape, dtype, _) = npy_meta(&path);
                let val = self.emit(OpKind::Input, vec![], shape, dtype);
                self.inputs.push((InputSource::Npy(path), val.id));
                TVal::Tensor(BVal { val, bdims: 0 })
            }
            "resize" => {
                if args.len() != 3 {
                    die(&format!("resize expects (image, height, width), got {} args", args.len()));
                }
                let v = self.trace(&args[0], env, fns).tensor("resize");
                let h = self.int_lit(&args[1], env, "resize height");
                let w = self.int_lit(&args[2], env, "resize width");
                self.resize_image(v, h, w)
            }
            "crop" => {
                if args.len() != 5 {
                    die(&format!("crop expects (image, top, left, height, width), got {} args", args.len()));
                }
                let v = self.trace(&args[0], env, fns).tensor("crop");
                let top = self.int_lit(&args[1], env, "crop top");
                let left = self.int_lit(&args[2], env, "crop left");
                let h = self.int_lit(&args[3], env, "crop height");
                let w = self.int_lit(&args[4], env, "crop width");
                self.crop_image(v, top, left, h, w)
            }
            "imshow" => {
                if args.len() != 1 {
                    die(&format!("imshow expects 1 arg, got {}", args.len()));
                }
                let v = self.trace(&args[0], env, fns);
                self.imshow(v)
            }
            "play" => {
                if args.len() != 1 {
                    die(&format!("play expects 1 arg, got {}", args.len()));
                }
                let v = self.trace(&args[0], env, fns);
                self.plan_play(&v);
                v
            }
            "uniform" => {
                if args.is_empty() {
                    die("uniform expects dimension literals");
                }
                let dims: Vec<usize> = args.iter().map(|a| self.int_lit(a, env, "dimension")).collect();
                let val = self.rng_uniform(&dims);
                TVal::Tensor(BVal { val, bdims: 0 })
            }
            "normal" => {
                if args.is_empty() {
                    die("normal expects dimension literals");
                }
                let dims: Vec<usize> = args.iter().map(|a| self.int_lit(a, env, "dimension")).collect();
                let u1 = self.rng_uniform(&dims);
                let u2 = self.rng_uniform(&dims);
                let floor_c = self.constant(1e-7, Dtype::F32);
                let safe = self.ewise("maximum", u1, floor_c);
                let ln = self.unary("log", &safe);
                let neg2 = self.constant(-2.0, Dtype::F32);
                let r2 = self.ewise("multiply", ln, neg2);
                let r = self.unary("sqrt", &r2);
                let tau = self.constant(std::f64::consts::TAU, Dtype::F32);
                let angle = self.ewise("multiply", u2, tau);
                let cosv = self.unary("cosine", &angle);
                let val = self.ewise("multiply", r, cosv);
                TVal::Tensor(BVal { val, bdims: 0 })
            }
            "dropout" => {
                if args.len() != 2 {
                    die(&format!("dropout expects (x, rate), got {} args", args.len()));
                }
                let x = self.trace(&args[0], env, fns).tensor("dropout");
                let rate = self.num_lit(&args[1], env, "dropout rate");
                if !(0.0..1.0).contains(&rate) {
                    die(&format!("dropout rate must be in [0, 1), got {}", rate));
                }
                if self.rng_baked {
                    return TVal::Tensor(x);
                }
                let u = self.rng_uniform(&x.val.shape.clone());
                let rate_c = self.constant(rate, Dtype::F32);
                let rate_b = self.broadcast(&rate_c, &u.shape.clone());
                let keep = self.compare("GE", &u, &rate_b);
                let scale = self.constant(1.0 / (1.0 - rate), x.val.dtype);
                let scaled = self.ewise("multiply", x.val.clone(), scale);
                let zero = self.zeros_like(&scaled);
                let val = self.select(&keep, &scaled, &zero);
                TVal::Tensor(BVal { val, bdims: x.bdims })
            }
            "sample" => {
                if args.len() != 1 {
                    die(&format!("sample expects 1 arg (logits), got {}", args.len()));
                }
                let logits = self.trace(&args[0], env, fns).tensor("sample");
                if per_shape(&logits).len() != 1 {
                    die(&format!("sample expects a logits vector, got shape {:?} (use vmap for batches)", per_shape(&logits)));
                }
                let shape = logits.val.shape.clone();
                let u = self.rng_uniform(&shape);
                let lo = self.constant(1e-7, Dtype::F32);
                let lo_b = self.broadcast(&lo, &shape);
                let hi = self.constant(1.0 - 1e-7, Dtype::F32);
                let hi_b = self.broadcast(&hi, &shape);
                let u = self.ewise("maximum", u, lo_b);
                let u = self.ewise("minimum", u, hi_b);
                let inner = self.unary("log", &u);
                let neg = self.unary("negate", &inner);
                let outer = self.unary("log", &neg);
                let gumbel = self.unary("negate", &outer);
                let noisy = self.ewise("add", logits.val.clone(), gumbel);
                self.arg_extreme(BVal { val: noisy, bdims: logits.bdims }, true)
            }
            "sort" | "argsort" | "argmax" | "argmin" | "cumsum" => {
                if args.len() != 1 {
                    die(&format!("{} expects 1 arg, got {}", name, args.len()));
                }
                let v = self.trace(&args[0], env, fns).tensor(name);
                if v.val.dtype == Dtype::I1 {
                    die(&format!("cannot {} booleans", name));
                }
                match name {
                    "sort" => self.sort_val(v),
                    "argsort" => self.argsort_val(v),
                    "argmax" => self.arg_extreme(v, true),
                    "argmin" => self.arg_extreme(v, false),
                    _ => self.cumsum_val(v),
                }
            }
            "one_hot" => {
                if args.len() != 2 {
                    die(&format!("one_hot expects (indices, depth), got {} args", args.len()));
                }
                let v = self.trace(&args[0], env, fns).tensor("one_hot");
                let n = self.int_lit(&args[1], env, "one_hot depth");
                self.one_hot_val(v, n)
            }
            "take" => {
                if args.len() != 2 {
                    die(&format!("take expects (values, indices), got {} args", args.len()));
                }
                let x = self.trace(&args[0], env, fns).tensor("take");
                let idx = self.trace(&args[1], env, fns).tensor("take indices");
                self.take_val(x, idx)
            }
            "transpose" => {
                if args.len() > 1 {
                    let v = self.trace(&args[0], env, fns).tensor("transpose");
                    let per = per_shape(&v);
                    let axes: Vec<usize> = args[1..].iter().map(|a| self.int_lit(a, env, "transpose axis")).collect();
                    let mut seen: Vec<usize> = axes.clone();
                    seen.sort();
                    if seen != (0..per.len()).collect::<Vec<usize>>() {
                        die(&format!("transpose axes {:?} must be a permutation of 0..{}", axes, per.len()));
                    }
                    let k = v.bdims;
                    let mut perm: Vec<usize> = (0..k).collect();
                    perm.extend(axes.iter().map(|a| k + a));
                    let mut shape: Vec<usize> = v.val.shape[..k].to_vec();
                    shape.extend(axes.iter().map(|&a| per[a]));
                    let val = self.emit(OpKind::Transpose(perm), vec![v.val.id], shape, v.val.dtype);
                    return TVal::Tensor(BVal { val, bdims: k });
                }
                if args.len() != 1 {
                    die(&format!("transpose expects 1 arg, got {}", args.len()));
                }
                let v = self.trace(&args[0], env, fns).tensor("transpose");
                let per = per_shape(&v);
                if per.len() != 2 {
                    die(&format!("transpose expects rank 2, got shape {:?}", per));
                }
                let k = v.bdims;
                let mut perm: Vec<usize> = (0..k).collect();
                perm.push(k + 1);
                perm.push(k);
                let mut shape: Vec<usize> = v.val.shape[..k].to_vec();
                shape.push(per[1]);
                shape.push(per[0]);
                let val = self.emit(OpKind::Transpose(perm), vec![v.val.id], shape, v.val.dtype);
                TVal::Tensor(BVal { val, bdims: k })
            }
            "matmul" => {
                if args.len() != 2 {
                    die(&format!("matmul expects 2 args, got {}", args.len()));
                }
                let a = self.trace(&args[0], env, fns).tensor("matmul");
                let b = self.trace(&args[1], env, fns).tensor("matmul");
                TVal::Tensor(self.bmatmul(a, b))
            }
            "grad" => {
                let (fname, target, traced) = if let Some(Expr::Field(obj, mname)) = args.first() {
                    let inst = self.trace(obj, env, fns);
                    let inst = self.barrier_tval(&inst);
                    let extras: Vec<TVal> = args[1..].iter().map(|a| self.trace(a, env, fns)).collect();
                    self.grad_depth += 1;
                    let out = self.call_method(inst.clone(), mname, extras, fns);
                    self.grad_depth -= 1;
                    (format!(".{}", mname), inst, out)
                } else {
                    let (fname, decl, vals) = self.transform_args("grad", args, env, fns);
                    let target = self.barrier_tval(&vals[0]);
                    let mut env2 = HashMap::new();
                    env2.insert(decl.params[0].clone(), target.clone());
                    for (param, v) in decl.params[1..].iter().zip(vals[1..].iter()) {
                        env2.insert(param.clone(), v.clone());
                    }
                    self.statics.push(HashMap::new());
                    self.grad_depth += 1;
                    let out = self.trace(&decl.body, &env2, fns);
                    self.grad_depth -= 1;
                    self.statics.pop();
                    (fname, target, out)
                };
                let y = match traced {
                    TVal::Tensor(b) => b,
                    TVal::Record(..) => die(&format!("grad requires a scalar-valued function; {} returned a record", fname)),
                };
                let per = per_shape(&y);
                if !per.is_empty() {
                    die(&format!("grad requires a scalar-valued function; {} returned shape {:?}", fname, per));
                }
                let seed = {
                    let one = self.constant(1.0, y.val.dtype);
                    if y.bdims > 0 { self.broadcast(&one, &y.val.shape.clone()) } else { one }
                };
                let mut leaves = Vec::new();
                collect_leaves(&target, &mut leaves);
                let targets: Vec<Val> = leaves.iter().map(|b| b.val.clone()).collect();
                let grads = self.backward(&y.val, &targets, seed);
                let mut grads = grads.into_iter();
                rebuild(&target, &mut grads)
            }
            "vmap" => {
                let (fname, decl, vals, frame) = if let Some(Expr::Field(obj, mname)) = args.first() {
                    let inst = self.trace(obj, env, fns);
                    let tag = match &inst {
                        TVal::Record(Some(t), _) => t.clone(),
                        _ => die("vmap of a method needs a module instance"),
                    };
                    let module = self.modules.get(&tag.module)
                        .unwrap_or_else(|| die(&format!("unknown module: {}", tag.module)))
                        .clone();
                    let decl = module.method(mname)
                        .unwrap_or_else(|| die(&format!("module {} has no method {}", tag.module, mname)))
                        .clone();
                    if args.len() - 1 != decl.params.len() - 1 {
                        die(&format!("vmap(.{}) expects {} args after the method, got {}",
                                     mname, decl.params.len() - 1, args.len() - 1));
                    }
                    let mut vals = vec![inst];
                    vals.extend(args[1..].iter().map(|a| self.trace(a, env, fns)));
                    let frame: HashMap<String, f64> = tag.statics.iter().cloned().collect();
                    (format!(".{}", mname), std::borrow::Cow::Owned(decl), vals, frame)
                } else {
                    let (fname, decl, vals) = self.transform_args("vmap", args, env, fns);
                    (fname, decl, vals, HashMap::new())
                };
                let mut k = 0;
                let mut prefix: Vec<usize> = Vec::new();
                let mut any_tensor = false;
                for v in &vals {
                    let mut leaves = Vec::new();
                    crate::batch::collect_leaves(v, &mut leaves);
                    if matches!(v, TVal::Tensor(_)) {
                        any_tensor = true;
                    }
                    for b in leaves {
                        if b.bdims > k {
                            k = b.bdims;
                            prefix = b.val.shape[..k].to_vec();
                        }
                    }
                }
                if !any_tensor {
                    die(&format!("vmap({}) needs a tensor argument to map over; records pass through unmapped", fname));
                }
                let lifted: Vec<TVal> = vals.iter().map(|v| match v {
                    TVal::Record(..) => v.clone(),
                    TVal::Tensor(b) => {
                        if b.bdims == k {
                            return v.clone();
                        }
                        let per: Vec<usize> = b.val.shape[b.bdims..].to_vec();
                        let mut target = prefix.clone();
                        target.extend(&per);
                        let mut dims: Vec<usize> = (0..b.bdims).collect();
                        dims.extend(k..k + per.len());
                        let val = self.broadcast_along(&b.val, &target, dims);
                        TVal::Tensor(BVal { val, bdims: k })
                    }
                }).collect();
                let mut n = None;
                for v in &lifted {
                    if let TVal::Tensor(b) = v {
                        let axis = match b.val.shape.get(k) {
                            Some(&a) => a,
                            None => die(&format!("vmap({}) arguments must have rank >= 1", fname)),
                        };
                        if *n.get_or_insert(axis) != axis {
                            die(&format!("vmap({}) arguments must share the mapped axis", fname));
                        }
                    }
                }
                let n = n.unwrap();
                let mut env2 = HashMap::new();
                for (param, v) in decl.params.iter().zip(&lifted) {
                    match v {
                        TVal::Tensor(b) => {
                            env2.insert(param.clone(), TVal::Tensor(BVal { val: b.val.clone(), bdims: k + 1 }));
                        }
                        TVal::Record(..) => {
                            env2.insert(param.clone(), v.clone());
                        }
                    }
                }
                self.statics.push(frame);
                let y = self.trace(&decl.body, &env2, fns).tensor("vmap function result");
                self.statics.pop();
                let val = if y.bdims == k + 1 {
                    y.val
                } else {
                    let per = per_shape(&y);
                    let mut target = prefix.clone();
                    target.push(n);
                    target.extend(&per);
                    let mut dims: Vec<usize> = (0..y.bdims).collect();
                    dims.extend(k + 1..k + 1 + per.len());
                    self.broadcast_along(&y.val, &target, dims)
                };
                TVal::Tensor(BVal { val, bdims: k })
            }
            "jacobian" => {
                let (fname, decl, mut vals) = self.transform_args("jacobian", args, env, fns);
                vals[0] = self.barrier_tval(&vals[0].clone());
                let mut env2 = HashMap::new();
                for (param, v) in decl.params.iter().zip(&vals) {
                    env2.insert(param.clone(), v.clone());
                }
                self.statics.push(HashMap::new());
                self.grad_depth += 1;
                let traced = self.trace(&decl.body, &env2, fns);
                self.grad_depth -= 1;
                self.statics.pop();
                let y = match traced {
                    TVal::Tensor(b) => b,
                    TVal::Record(..) => die(&format!("jacobian requires a vector-valued function; {} returned a record", fname)),
                };
                let per = per_shape(&y);
                if per.len() != 1 {
                    die(&format!("jacobian requires a vector-valued function; {} returned shape {:?} (use grad for scalars)", fname, per));
                }
                let m = per[0];
                let x = vals[0].clone().tensor("jacobian target");
                let mut row_shape: Vec<usize> = x.val.shape[..x.bdims].to_vec();
                row_shape.push(1);
                row_shape.extend(&x.val.shape[x.bdims..]);
                let mut rows = Vec::new();
                for i in 0..m {
                    let mut hots = Vec::new();
                    for j in 0..m {
                        let c = self.constant(if i == j { 1.0 } else { 0.0 }, y.val.dtype);
                        hots.push(BVal { val: c, bdims: 0 });
                    }
                    let hot = self.stack(hots);
                    let seed = if y.bdims == 0 {
                        hot.val
                    } else {
                        self.broadcast_along(&hot.val, &y.val.shape.clone(), vec![y.bdims])
                    };
                    let row = self.backward(&y.val, &[x.val.clone()], seed).remove(0);
                    rows.push(self.reshape(&row, row_shape.clone()).id);
                }
                let mut shape: Vec<usize> = x.val.shape[..x.bdims].to_vec();
                shape.push(m);
                shape.extend(&x.val.shape[x.bdims..]);
                let val = if rows.len() == 1 {
                    self.val(rows[0])
                } else {
                    self.emit(OpKind::Concat(x.bdims), rows, shape, x.val.dtype)
                };
                TVal::Tensor(BVal { val, bdims: x.bdims })
            }
            _ => die(&format!("undefined function: {}", name)),
        }
    }

    pub fn transform_args<'f>(&mut self, transform: &str, args: &[Expr], env: &HashMap<String, TVal>, fns: &'f HashMap<String, Decl>) -> (String, std::borrow::Cow<'f, Decl>, Vec<TVal>) {
        let fname = match args.first() {
            Some(Expr::Var(s)) => s.clone(),
            _ => die(&format!("{} expects a function name as its first argument", transform)),
        };
        let decl = match fns.get(&fname) {
            Some(decl) => std::borrow::Cow::Borrowed(decl),
            None => {
                let params: Vec<String> = (0..args.len() - 1).map(|i| format!("__arg{}", i)).collect();
                let body = Expr::Call(fname.clone(), params.iter().map(|p| Expr::Var(p.clone())).collect());
                std::borrow::Cow::Owned(Decl { params, body })
            }
        };
        if args.len() - 1 != decl.params.len() {
            die(&format!("{}({}) expects {} args after the function name, got {}",
                         transform, fname, decl.params.len(), args.len() - 1));
        }
        let vals: Vec<TVal> = args[1..].iter().map(|a| self.trace(a, env, fns)).collect();
        (fname, decl, vals)
    }
}

