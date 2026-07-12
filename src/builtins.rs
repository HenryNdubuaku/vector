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
                if start.bdims != 0 || !start.val.shape.is_empty() {
                    die("slice start must be a scalar");
                }
                let size = self.int_lit(&args[2], env, "slice size");
                if size == 0 || size > x.val.shape[0] {
                    die(&format!("slice size {} out of range for leading dim {}", size, x.val.shape[0]));
                }
                let idx = self.convert(&start.val, Dtype::I64);
                let mut inputs = vec![x.val.id, idx.id];
                for _ in 1..x.val.shape.len() {
                    inputs.push(self.constant(0.0, Dtype::I64).id);
                }
                let mut sizes = vec![size];
                sizes.extend(&x.val.shape[1..]);
                let shape = sizes.clone();
                let val = self.emit(OpKind::DynSlice(sizes), inputs, shape, x.val.dtype);
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
            "transpose" => {
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
                let (fname, decl, vals) = self.transform_args("vmap", args, env, fns);
                let vals: Vec<BVal> = vals.into_iter().map(|v| v.tensor("vmap argument")).collect();
                let k = vals[0].bdims;
                if vals.iter().any(|v| v.bdims != k) {
                    die(&format!("vmap({}) arguments must come from the same batching depth; inline constant arguments into the function body", fname));
                }
                let n = match vals[0].val.shape.get(k) {
                    Some(&n) => n,
                    None => die(&format!("vmap({}) arguments must have rank >= 1", fname)),
                };
                for v in &vals {
                    if v.val.shape.get(k) != Some(&n) {
                        die(&format!("vmap({}) arguments must share the mapped axis: {:?} vs {:?}",
                                     fname, vals[0].val.shape, v.val.shape));
                    }
                }
                let mut env2 = HashMap::new();
                for (param, v) in decl.params.iter().zip(&vals) {
                    env2.insert(param.clone(), TVal::Tensor(BVal { val: v.val.clone(), bdims: k + 1 }));
                }
                self.statics.push(HashMap::new());
                let y = self.trace(&decl.body, &env2, fns).tensor("vmap function result");
                self.statics.pop();
                let val = if y.bdims == k + 1 {
                    y.val
                } else {
                    let per = per_shape(&y);
                    let mut target: Vec<usize> = vals[0].val.shape[..k].to_vec();
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

    pub fn transform_args<'f>(&mut self, transform: &str, args: &[Expr], env: &HashMap<String, TVal>, fns: &'f HashMap<String, Decl>) -> (String, &'f Decl, Vec<TVal>) {
        let fname = match args.first() {
            Some(Expr::Var(s)) => s.clone(),
            _ => die(&format!("{} expects a function name as its first argument", transform)),
        };
        let decl = fns.get(&fname)
            .unwrap_or_else(|| die(&format!("undefined function: {}", fname)));
        if args.len() - 1 != decl.params.len() {
            die(&format!("{}({}) expects {} args after the function name, got {}",
                         transform, fname, decl.params.len(), args.len() - 1));
        }
        let vals: Vec<TVal> = args[1..].iter().map(|a| self.trace(a, env, fns)).collect();
        (fname, decl, vals)
    }
}

