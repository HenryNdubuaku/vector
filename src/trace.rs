use std::collections::HashMap;

use crate::die;
use crate::graph::{broadcast_shape, per_shape, BVal, Dtype, Node, OpKind, Val};
use crate::npy::npy_meta;
use crate::parser::{Decl, Expr, Op};

pub struct Tracer {
    pub nodes: Vec<Node>,
    pub prints: Vec<Val>,
    pub inputs: Vec<(String, usize)>,
}

impl Tracer {
    pub fn emit(&mut self, kind: OpKind, inputs: Vec<usize>, shape: Vec<usize>, dtype: Dtype) -> Val {
        let id = self.nodes.len();
        self.nodes.push(Node { kind, inputs, shape: shape.clone(), dtype });
        Val { id, shape, dtype }
    }

    pub fn val(&self, id: usize) -> Val {
        let node = &self.nodes[id];
        Val { id, shape: node.shape.clone(), dtype: node.dtype }
    }

    pub fn constant(&mut self, n: f64, dtype: Dtype) -> Val {
        self.emit(OpKind::Constant(n), vec![], vec![], dtype)
    }

    pub fn convert(&mut self, v: &Val, dtype: Dtype) -> Val {
        if v.dtype == dtype {
            return v.clone();
        }
        self.emit(OpKind::Convert, vec![v.id], v.shape.clone(), dtype)
    }

    pub fn broadcast(&mut self, v: &Val, shape: &[usize]) -> Val {
        let offset = shape.len() - v.shape.len();
        let dims: Vec<usize> = (offset..shape.len()).collect();
        self.broadcast_along(v, shape, dims)
    }

    pub fn broadcast_along(&mut self, v: &Val, shape: &[usize], dims: Vec<usize>) -> Val {
        self.emit(OpKind::Broadcast(dims), vec![v.id], shape.to_vec(), v.dtype)
    }

    pub fn unary(&mut self, name: &str, v: &Val) -> Val {
        self.emit(OpKind::Unary(name.to_string()), vec![v.id], v.shape.clone(), v.dtype)
    }

    pub fn reshape(&mut self, v: &Val, shape: Vec<usize>) -> Val {
        self.emit(OpKind::Reshape, vec![v.id], shape, v.dtype)
    }

    pub fn compare(&mut self, dir: &str, a: &Val, b: &Val) -> Val {
        self.emit(OpKind::Compare(dir.to_string()), vec![a.id, b.id], a.shape.clone(), Dtype::I1)
    }

    pub fn select(&mut self, pred: &Val, on_true: &Val, on_false: &Val) -> Val {
        self.emit(OpKind::Select, vec![pred.id, on_true.id, on_false.id], on_true.shape.clone(), on_true.dtype)
    }

    pub fn zeros(&mut self, shape: &[usize], dtype: Dtype) -> Val {
        let zero = self.constant(0.0, dtype);
        if shape.is_empty() { zero } else { self.broadcast(&zero, shape) }
    }

    pub fn zeros_like(&mut self, v: &Val) -> Val {
        self.zeros(&v.shape.clone(), v.dtype)
    }

    pub fn ewise(&mut self, name: &str, a: Val, b: Val) -> Val {
        let (a, b) = if a.dtype == b.dtype {
            (a, b)
        } else if a.shape.is_empty() {
            (self.convert(&a, b.dtype), b)
        } else if b.shape.is_empty() {
            let b = self.convert(&b, a.dtype);
            (a, b)
        } else {
            die(&format!("dtype mismatch: {} vs {}", a.dtype.name(), b.dtype.name()));
        };
        let (a, b) = if a.shape == b.shape {
            (a, b)
        } else if a.shape.len() <= b.shape.len() && b.shape.ends_with(&a.shape) {
            (self.broadcast(&a, &b.shape.clone()), b)
        } else if b.shape.len() < a.shape.len() && a.shape.ends_with(&b.shape) {
            let b = self.broadcast(&b, &a.shape.clone());
            (a, b)
        } else {
            die(&format!("shape mismatch: {:?} vs {:?} (broadcast aligns trailing dims)", a.shape, b.shape));
        };
        let (shape, dtype) = (a.shape.clone(), a.dtype);
        self.emit(OpKind::Ewise(name.to_string()), vec![a.id, b.id], shape, dtype)
    }

    fn binop(&mut self, op: Op, a: BVal, b: BVal) -> BVal {
        let name = match op {
            Op::Add => "add",
            Op::Sub => "subtract",
            Op::Mul => "multiply",
            Op::Div => "divide",
        };
        self.bewise(name, a, b)
    }

    pub fn dot(&mut self, a: &Val, b: &Val, lb: Vec<usize>, rb: Vec<usize>, lc: Vec<usize>, rc: Vec<usize>) -> Val {
        let mut shape: Vec<usize> = lb.iter().map(|&i| a.shape[i]).collect();
        shape.extend(a.shape.iter().enumerate()
            .filter(|(i, _)| !lb.contains(i) && !lc.contains(i))
            .map(|(_, &d)| d));
        shape.extend(b.shape.iter().enumerate()
            .filter(|(i, _)| !rb.contains(i) && !rc.contains(i))
            .map(|(_, &d)| d));
        self.emit(OpKind::Dot(lb, rb, lc, rc), vec![a.id, b.id], shape, a.dtype)
    }

    fn align(&mut self, v: &BVal, prefix: &[usize], per: &[usize]) -> Val {
        let mut target = prefix.to_vec();
        target.extend(per);
        if v.val.shape == target {
            return v.val.clone();
        }
        let offset = per.len() - per_shape(v).len();
        let mut dims: Vec<usize> = (0..v.bdims).collect();
        dims.extend(prefix.len() + offset..prefix.len() + per.len());
        self.broadcast_along(&v.val, &target, dims)
    }

    fn batch_prefix(a: &BVal, b: &BVal) -> Vec<usize> {
        let deep = if a.bdims >= b.bdims { a } else { b };
        deep.val.shape[..deep.bdims].to_vec()
    }

    fn balign(&mut self, a: BVal, b: BVal) -> (Val, Val, Vec<usize>, Vec<usize>) {
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
        let per = broadcast_shape(&per_shape(&a), &per_shape(&b));
        let prefix = Self::batch_prefix(&a, &b);
        let av = self.align(&a, &prefix, &per);
        let bv = self.align(&b, &prefix, &per);
        (av, bv, prefix, per)
    }

    fn bewise(&mut self, name: &str, a: BVal, b: BVal) -> BVal {
        if a.val.dtype == Dtype::I1 || b.val.dtype == Dtype::I1 {
            die(&format!("arithmetic on booleans: {}", name));
        }
        let (av, bv, prefix, per) = self.balign(a, b);
        let mut shape = prefix.clone();
        shape.extend(&per);
        let dtype = av.dtype;
        let val = self.emit(OpKind::Ewise(name.to_string()), vec![av.id, bv.id], shape, dtype);
        BVal { val, bdims: prefix.len() }
    }

    fn bcompare(&mut self, dir: &str, a: BVal, b: BVal) -> BVal {
        if a.val.dtype == Dtype::I1 || b.val.dtype == Dtype::I1 {
            die("cannot compare booleans");
        }
        let (av, bv, prefix, per) = self.balign(a, b);
        let mut shape = prefix.clone();
        shape.extend(&per);
        let val = self.emit(OpKind::Compare(dir.to_string()), vec![av.id, bv.id], shape, Dtype::I1);
        BVal { val, bdims: prefix.len() }
    }

    fn bunary(&mut self, name: &str, v: &BVal) -> BVal {
        BVal { val: self.unary(name, &v.val), bdims: v.bdims }
    }

    fn bmatmul(&mut self, a: BVal, b: BVal) -> BVal {
        if a.val.dtype != b.val.dtype {
            die(&format!("matmul dtype mismatch: {} vs {}", a.val.dtype.name(), b.val.dtype.name()));
        }
        let pa = per_shape(&a);
        let pb = per_shape(&b);
        if pa.is_empty() || pa.len() > 2 || pb.is_empty() || pb.len() > 2 {
            die(&format!("matmul supports rank 1 and 2, got {:?} vs {:?}", pa, pb));
        }
        if pa[pa.len() - 1] != pb[0] {
            die(&format!("matmul contraction mismatch: {:?} vs {:?}", pa, pb));
        }
        if a.bdims == 0 && b.bdims == 0 {
            let val = self.dot(&a.val, &b.val, vec![], vec![], vec![pa.len() - 1], vec![0]);
            return BVal { val, bdims: 0 };
        }
        let prefix = Self::batch_prefix(&a, &b);
        let k = prefix.len();
        let av = self.align(&a, &prefix, &pa);
        let bv = self.align(&b, &prefix, &pb);
        let batch: Vec<usize> = (0..k).collect();
        let val = self.dot(&av, &bv, batch.clone(), batch, vec![k + pa.len() - 1], vec![k]);
        BVal { val, bdims: k }
    }

    pub fn reduce_sum(&mut self, v: &Val, axes: &[usize]) -> Val {
        if axes.is_empty() {
            return v.clone();
        }
        let init = self.constant(0.0, v.dtype);
        let out_shape: Vec<usize> = v.shape.iter().enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, &d)| d)
            .collect();
        self.emit(OpKind::Reduce(axes.to_vec()), vec![v.id, init.id], out_shape, v.dtype)
    }

    fn stack(&mut self, vals: Vec<BVal>) -> BVal {
        let inner_shape = per_shape(&vals[0]);
        let dtype = vals[0].val.dtype;
        for v in &vals[1..] {
            if per_shape(v) != inner_shape {
                die(&format!("array literal has inconsistent shapes: {:?} vs {:?}", inner_shape, per_shape(v)));
            }
            if v.val.dtype != dtype {
                die(&format!("array literal has inconsistent dtypes: {} vs {}", dtype.name(), v.val.dtype.name()));
            }
        }
        let deepest = vals.iter().max_by_key(|v| v.bdims).unwrap();
        let prefix: Vec<usize> = deepest.val.shape[..deepest.bdims].to_vec();
        let k = prefix.len();
        let mut row_shape = prefix.clone();
        row_shape.push(1);
        row_shape.extend(&inner_shape);
        let mut rows = Vec::new();
        for v in &vals {
            let aligned = self.align(v, &prefix, &inner_shape);
            rows.push(self.reshape(&aligned, row_shape.clone()));
        }
        if rows.len() == 1 {
            return BVal { val: rows.into_iter().next().unwrap(), bdims: k };
        }
        let mut shape = prefix;
        shape.push(rows.len());
        shape.extend(&inner_shape);
        let ids: Vec<usize> = rows.iter().map(|r| r.id).collect();
        let val = self.emit(OpKind::Concat(k), ids, shape, dtype);
        BVal { val, bdims: k }
    }

    pub fn trace(&mut self, e: &Expr, env: &HashMap<String, BVal>, fns: &HashMap<String, Decl>) -> BVal {
        match e {
            Expr::Num(n) => {
                let val = self.constant(*n, Dtype::F64);
                BVal { val, bdims: 0 }
            }
            Expr::Str(_) => die("string literals are only valid as the argument of load"),
            Expr::Neg(inner) => {
                let v = self.trace(inner, env, fns);
                self.bunary("negate", &v)
            }
            Expr::Arr(elems) => {
                if elems.is_empty() {
                    die("empty array literal");
                }
                let vals: Vec<BVal> = elems.iter().map(|el| self.trace(el, env, fns)).collect();
                self.stack(vals)
            }
            Expr::Var(s) => {
                if let Some(v) = env.get(s) {
                    v.clone()
                } else if fns.contains_key(s) {
                    die(&format!("'{}' is a function; first-class functions aren't supported yet", s));
                } else {
                    die(&format!("undefined: {}", s));
                }
            }
            Expr::Bin(op, l, r) => {
                let l = self.trace(l, env, fns);
                let r = self.trace(r, env, fns);
                self.binop(*op, l, r)
            }
            Expr::Cmp(dir, l, r) => {
                let l = self.trace(l, env, fns);
                let r = self.trace(r, env, fns);
                self.bcompare(dir, l, r)
            }
            Expr::For(var, start, end, stmts, rest) => {
                let mut env2 = env.clone();
                for k in *start..*end {
                    let kv = self.constant(k as f64, Dtype::F64);
                    env2.insert(var.clone(), BVal { val: kv, bdims: 0 });
                    for (name, stmt) in stmts {
                        let v = self.trace(stmt, &env2, fns);
                        if let Some(name) = name {
                            env2.insert(name.clone(), v);
                        }
                    }
                }
                match env.get(var) {
                    Some(orig) => env2.insert(var.clone(), orig.clone()),
                    None => env2.remove(var),
                };
                self.trace(rest, &env2, fns)
            }
            Expr::Let(name, value, body) => {
                let v = self.trace(value, env, fns);
                let mut env2 = env.clone();
                env2.insert(name.clone(), v);
                self.trace(body, &env2, fns)
            }
            Expr::Seq(first, rest) => {
                self.trace(first, env, fns);
                self.trace(rest, env, fns)
            }
            Expr::Call(name, args) => {
                if let Some(decl) = fns.get(name) {
                    if decl.params.len() != args.len() {
                        die(&format!("arity mismatch: {} expects {} args, got {}",
                                     name, decl.params.len(), args.len()));
                    }
                    let mut env2 = HashMap::new();
                    for (param, arg) in decl.params.iter().zip(args.iter()) {
                        env2.insert(param.clone(), self.trace(arg, env, fns));
                    }
                    self.trace(&decl.body, &env2, fns)
                } else {
                    self.builtin(name, args, env, fns)
                }
            }
        }
    }

    fn builtin(&mut self, name: &str, args: &[Expr], env: &HashMap<String, BVal>, fns: &HashMap<String, Decl>) -> BVal {
        match name {
            "sum" | "mean" => {
                if args.is_empty() || args.len() > 2 {
                    die(&format!("{} expects 1 or 2 args, got {}", name, args.len()));
                }
                let v = self.trace(&args[0], env, fns);
                let per = per_shape(&v);
                let per_axes: Vec<usize> = if args.len() == 2 {
                    vec![axis_arg(&args[1], &per)]
                } else {
                    (0..per.len()).collect()
                };
                let axes: Vec<usize> = per_axes.iter().map(|a| a + v.bdims).collect();
                let total = self.reduce_sum(&v.val, &axes);
                let total = BVal { val: total, bdims: v.bdims };
                if name == "sum" {
                    return total;
                }
                let count: usize = per_axes.iter().map(|&d| per[d]).product();
                let denom = self.constant(count as f64, v.val.dtype);
                self.bewise("divide", total, BVal { val: denom, bdims: 0 })
            }
            "print" => {
                if args.len() != 1 {
                    die(&format!("print expects 1 arg, got {}", args.len()));
                }
                let v = self.trace(&args[0], env, fns);
                if v.val.dtype == Dtype::I1 {
                    die("cannot print booleans; use where to select values");
                }
                self.prints.push(v.val.clone());
                v
            }
            "where" => {
                if args.len() != 3 {
                    die(&format!("where expects 3 args (condition, then, else), got {}", args.len()));
                }
                let c = self.trace(&args[0], env, fns);
                if c.val.dtype != Dtype::I1 {
                    die("where condition must be a comparison");
                }
                let a = self.trace(&args[1], env, fns);
                let b = self.trace(&args[2], env, fns);
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
                BVal { val, bdims: prefix.len() }
            }
            "f32" | "f64" => {
                if args.len() != 1 {
                    die(&format!("{} expects 1 arg, got {}", name, args.len()));
                }
                let dtype = if name == "f32" { Dtype::F32 } else { Dtype::F64 };
                let v = self.trace(&args[0], env, fns);
                let val = self.convert(&v.val, dtype);
                BVal { val, bdims: v.bdims }
            }
            "exp" | "log" | "tanh" | "sqrt" => {
                if args.len() != 1 {
                    die(&format!("{} expects 1 arg, got {}", name, args.len()));
                }
                let v = self.trace(&args[0], env, fns);
                let op = if name == "exp" { "exponential" } else { name };
                self.bunary(op, &v)
            }
            "max" | "min" => {
                if args.len() != 2 {
                    die(&format!("{} expects 2 args, got {}", name, args.len()));
                }
                let a = self.trace(&args[0], env, fns);
                let b = self.trace(&args[1], env, fns);
                let op = if name == "max" { "maximum" } else { "minimum" };
                self.bewise(op, a, b)
            }
            "load" => {
                if args.len() != 1 {
                    die(&format!("load expects 1 arg, got {}", args.len()));
                }
                let path = match &args[0] {
                    Expr::Str(s) => s.clone(),
                    _ => die("load expects a file path string literal"),
                };
                if let Some(&(_, id)) = self.inputs.iter().find(|(p, _)| *p == path) {
                    return BVal { val: self.val(id), bdims: 0 };
                }
                let (shape, dtype, _) = npy_meta(&path);
                let val = self.emit(OpKind::Input, vec![], shape, dtype);
                self.inputs.push((path, val.id));
                BVal { val, bdims: 0 }
            }
            "matmul" => {
                if args.len() != 2 {
                    die(&format!("matmul expects 2 args, got {}", args.len()));
                }
                let a = self.trace(&args[0], env, fns);
                let b = self.trace(&args[1], env, fns);
                self.bmatmul(a, b)
            }
            "grad" => {
                let (fname, decl, vals) = self.transform_args("grad", args, env, fns);
                let mut env2 = HashMap::new();
                for (param, v) in decl.params.iter().zip(&vals) {
                    env2.insert(param.clone(), v.clone());
                }
                let y = self.trace(&decl.body, &env2, fns);
                let per = per_shape(&y);
                if !per.is_empty() {
                    die(&format!("grad requires a scalar-valued function; {} returned shape {:?}", fname, per));
                }
                let seed = {
                    let one = self.constant(1.0, y.val.dtype);
                    if y.bdims > 0 { self.broadcast(&one, &y.val.shape.clone()) } else { one }
                };
                let val = self.backward(&y.val, &vals[0].val, seed);
                BVal { val, bdims: vals[0].bdims }
            }
            "vmap" => {
                let (fname, decl, vals) = self.transform_args("vmap", args, env, fns);
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
                    env2.insert(param.clone(), BVal { val: v.val.clone(), bdims: k + 1 });
                }
                let y = self.trace(&decl.body, &env2, fns);
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
                BVal { val, bdims: k }
            }
            "jacobian" => {
                let (fname, decl, vals) = self.transform_args("jacobian", args, env, fns);
                let mut env2 = HashMap::new();
                for (param, v) in decl.params.iter().zip(&vals) {
                    env2.insert(param.clone(), v.clone());
                }
                let y = self.trace(&decl.body, &env2, fns);
                let per = per_shape(&y);
                if per.len() != 1 {
                    die(&format!("jacobian requires a vector-valued function; {} returned shape {:?} (use grad for scalars)", fname, per));
                }
                let m = per[0];
                let x = vals[0].clone();
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
                    let row = self.backward(&y.val, &x.val, seed);
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
                BVal { val, bdims: x.bdims }
            }
            _ => die(&format!("undefined function: {}", name)),
        }
    }

    fn transform_args<'f>(&mut self, transform: &str, args: &[Expr], env: &HashMap<String, BVal>, fns: &'f HashMap<String, Decl>) -> (String, &'f Decl, Vec<BVal>) {
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
        let vals: Vec<BVal> = args[1..].iter().map(|a| self.trace(a, env, fns)).collect();
        (fname, decl, vals)
    }
}

fn axis_arg(e: &Expr, shape: &[usize]) -> usize {
    let n = match e {
        Expr::Num(n) => *n,
        _ => die("reduction axis must be a number literal"),
    };
    if n.fract() != 0.0 || n < 0.0 || n as usize >= shape.len() {
        die(&format!("reduction axis {} out of range for shape {:?}", n, shape));
    }
    n as usize
}
