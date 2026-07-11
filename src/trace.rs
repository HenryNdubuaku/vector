use std::collections::{HashMap, HashSet};

use crate::die;
use crate::graph::{broadcast_shape, per_shape, BVal, Dtype, ModTag, Node, OpKind, TVal, Val};
use crate::npy::npy_meta;
use crate::parser::{Decl, Expr, ModuleDecl, Op};

pub struct Tracer {
    pub nodes: Vec<Node>,
    pub prints: Vec<(Option<String>, Val)>,
    pub inputs: Vec<(String, usize)>,
    pub modules: HashMap<String, ModuleDecl>,
    pub statics: Vec<HashMap<String, f64>>,
    pub rng: u64,
    pub claimed: HashSet<usize>,
    pub region_depth: usize,
    pub grad_depth: usize,
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

    fn binop(&mut self, op: Op, a: TVal, b: TVal) -> TVal {
        let name = match op {
            Op::Add => "add",
            Op::Sub => "subtract",
            Op::Mul => "multiply",
            Op::Div => "divide",
        };
        self.tmap2(name, a, b)
    }

    fn tmap2(&mut self, name: &str, a: TVal, b: TVal) -> TVal {
        match (a, b) {
            (TVal::Tensor(x), TVal::Tensor(y)) => TVal::Tensor(self.bewise(name, x, y)),
            (TVal::Record(ta, ra), TVal::Record(tb, rb)) => {
                if ra.len() != rb.len() {
                    die(&format!("record fields mismatch: {} vs {} fields", ra.len(), rb.len()));
                }
                let mut fields = Vec::new();
                for (k, av) in ra {
                    let bv = rb.iter().find(|(k2, _)| *k2 == k).map(|(_, v)| v.clone())
                        .unwrap_or_else(|| die(&format!("record fields mismatch: missing field '{}'", k)));
                    let r = self.tmap2(name, av, bv);
                    fields.push((k, r));
                }
                TVal::Record(ta.or(tb), fields)
            }
            (TVal::Record(ta, ra), b) => {
                let mut fields = Vec::new();
                for (k, av) in ra {
                    let r = self.tmap2(name, av, b.clone());
                    fields.push((k, r));
                }
                TVal::Record(ta, fields)
            }
            (a, TVal::Record(tb, rb)) => {
                let mut fields = Vec::new();
                for (k, bv) in rb {
                    let r = self.tmap2(name, a.clone(), bv);
                    fields.push((k, r));
                }
                TVal::Record(tb, fields)
            }
        }
    }

    fn tunary(&mut self, name: &str, v: &TVal) -> TVal {
        match v {
            TVal::Tensor(b) => TVal::Tensor(self.bunary(name, b)),
            TVal::Record(tag, fields) => {
                let mut out = Vec::new();
                for (k, f) in fields {
                    let r = self.tunary(name, f);
                    out.push((k.clone(), r));
                }
                TVal::Record(tag.clone(), out)
            }
        }
    }

    fn tconvert(&mut self, v: &TVal, dtype: Dtype) -> TVal {
        match v {
            TVal::Tensor(b) => {
                let val = self.convert(&b.val, dtype);
                TVal::Tensor(BVal { val, bdims: b.bdims })
            }
            TVal::Record(tag, fields) => {
                let mut out = Vec::new();
                for (k, f) in fields {
                    let r = self.tconvert(f, dtype);
                    out.push((k.clone(), r));
                }
                TVal::Record(tag.clone(), out)
            }
        }
    }

    fn push_prints(&mut self, label: Option<String>, v: &TVal) {
        match v {
            TVal::Tensor(b) => self.prints.push((label, b.val.clone())),
            TVal::Record(_, fields) => {
                for (k, f) in fields {
                    let path = match &label {
                        Some(p) => format!("{}.{}", p, k),
                        None => k.clone(),
                    };
                    self.push_prints(Some(path), f);
                }
            }
        }
    }

    fn instantiate(&mut self, name: &str, args: &[Expr], env: &HashMap<String, TVal>, fns: &HashMap<String, Decl>) -> TVal {
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

    fn call_method(&mut self, callee: TVal, method: &str, args: Vec<TVal>, fns: &HashMap<String, Decl>) -> TVal {
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

    pub fn reduce(&mut self, reducer: &str, init: f64, v: &Val, axes: &[usize]) -> Val {
        if axes.is_empty() {
            return v.clone();
        }
        let init = self.constant(init, v.dtype);
        let out_shape: Vec<usize> = v.shape.iter().enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, &d)| d)
            .collect();
        self.emit(OpKind::Reduce(reducer.to_string(), axes.to_vec()), vec![v.id, init.id], out_shape, v.dtype)
    }

    pub fn reduce_sum(&mut self, v: &Val, axes: &[usize]) -> Val {
        self.reduce("add", 0.0, v, axes)
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

    pub fn trace(&mut self, e: &Expr, env: &HashMap<String, TVal>, fns: &HashMap<String, Decl>) -> TVal {
        match e {
            Expr::Num(n) => {
                let val = self.constant(*n, Dtype::F64);
                TVal::Tensor(BVal { val, bdims: 0 })
            }
            Expr::Str(_) => die("string literals are only valid as the argument of load"),
            Expr::RecordLit(fields) => {
                let mut out = Vec::new();
                for (k, v) in fields {
                    let t = self.trace(v, env, fns);
                    out.push((k.clone(), t));
                }
                TVal::Record(None, out)
            }
            Expr::Field(inner, name) => match self.trace(inner, env, fns) {
                TVal::Record(_, fields) => fields.iter()
                    .find(|(k, _)| k == name)
                    .map(|(_, v)| v.clone())
                    .unwrap_or_else(|| {
                        let keys: Vec<&str> = fields.iter().map(|(k, _)| k.as_str()).collect();
                        die(&format!("record has no field '{}' (fields: {})", name, keys.join(", ")))
                    }),
                TVal::Tensor(_) => die(&format!("field access '.{}' on a tensor", name)),
            },
            Expr::Apply(callee, args) => {
                if let Expr::Field(obj, mname) = callee.as_ref() {
                    let o = self.trace(obj, env, fns);
                    let argv: Vec<TVal> = args.iter().map(|a| self.trace(a, env, fns)).collect();
                    if let TVal::Record(tag, fields) = &o {
                        if let Some((_, field_val)) = fields.iter().find(|(k, _)| k == mname) {
                            let field_val = field_val.clone();
                            return self.call_method(field_val, "forward", argv, fns);
                        }
                        if let Some(t) = tag {
                            let has_method = self.modules.get(&t.module)
                                .map(|m| m.method(mname).is_some())
                                .unwrap_or(false);
                            if has_method {
                                return self.call_method(o, mname, argv, fns);
                            }
                        }
                    }
                    die(&format!("no field or method '{}' to call", mname));
                }
                let c = self.trace(callee, env, fns);
                let argv: Vec<TVal> = args.iter().map(|a| self.trace(a, env, fns)).collect();
                self.call_method(c, "forward", argv, fns)
            }
            Expr::Neg(inner) => {
                let v = self.trace(inner, env, fns);
                self.tunary("negate", &v)
            }
            Expr::Arr(elems) => {
                if elems.is_empty() {
                    die("empty array literal");
                }
                let vals: Vec<BVal> = elems.iter()
                    .map(|el| self.trace(el, env, fns).tensor("array literal element"))
                    .collect();
                TVal::Tensor(self.stack(vals))
            }
            Expr::Var(s) => {
                if let Some(v) = env.get(s) {
                    v.clone()
                } else if let Some(n) = self.static_num(s).or_else(|| named_const(s)) {
                    let val = self.constant(n, Dtype::F64);
                    TVal::Tensor(BVal { val, bdims: 0 })
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
                let l = self.trace(l, env, fns).tensor("comparison operand");
                let r = self.trace(r, env, fns).tensor("comparison operand");
                TVal::Tensor(self.bcompare(dir, l, r))
            }
            Expr::For(var, start_e, end_e, stmts, rest) => {
                let start = self.int_lit(start_e, env, "range start");
                let end = self.int_lit(end_e, env, "range end");
                let mut carried: Vec<String> = Vec::new();
                for (name, _) in stmts {
                    if let Some(n) = name {
                        if n == var {
                            die(&format!("cannot assign to loop variable {}", var));
                        }
                        if env.contains_key(n) && !carried.contains(n) {
                            carried.push(n.clone());
                        }
                    }
                }
                if self.grad_depth > 0 {
                    let mut env2 = env.clone();
                    self.region_depth += 1;
                    for k in start..end {
                        let kv = self.constant(k as f64, Dtype::F64);
                        env2.insert(var.clone(), TVal::Tensor(BVal { val: kv, bdims: 0 }));
                        for (name, stmt) in stmts {
                            let v = self.trace(stmt, &env2, fns);
                            if let Some(name) = name {
                                env2.insert(name.clone(), v);
                            }
                        }
                    }
                    self.region_depth -= 1;
                    let mut env3 = env.clone();
                    for name in &carried {
                        env3.insert(name.clone(), env2[name].clone());
                    }
                    return self.trace(rest, &env3, fns);
                }
                let init_vals: Vec<TVal> = carried.iter().map(|n| env[n].clone()).collect();
                let init_leaves_per: Vec<Vec<BVal>> = init_vals.iter().map(|v| {
                    let mut l = Vec::new();
                    collect_leaves(v, &mut l);
                    l
                }).collect();

                let limit = self.constant(end as f64, Dtype::F64);
                let counter_init = self.constant(start as f64, Dtype::F64);

                let body_start = self.nodes.len();
                let counter_arg = self.emit(OpKind::IterArg, vec![], vec![], Dtype::F64);
                let arg_leaves: Vec<BVal> = init_leaves_per.iter().flatten().map(|b| {
                    let val = self.emit(OpKind::IterArg, vec![], b.val.shape.clone(), b.val.dtype);
                    BVal { val, bdims: b.bdims }
                }).collect();

                let mut env2 = env.clone();
                env2.insert(var.clone(), TVal::Tensor(BVal { val: counter_arg.clone(), bdims: 0 }));
                let mut arg_iter = arg_leaves.iter().map(|b| b.val.clone()).collect::<Vec<_>>().into_iter();
                for (name, structure) in carried.iter().zip(&init_vals) {
                    env2.insert(name.clone(), rebuild(structure, &mut arg_iter));
                }

                self.region_depth += 1;
                for (name, stmt) in stmts {
                    let v = self.trace(stmt, &env2, fns);
                    if let Some(name) = name {
                        env2.insert(name.clone(), v);
                    }
                }
                self.region_depth -= 1;

                let one = self.constant(1.0, Dtype::F64);
                let next_counter = self.ewise("add", counter_arg.clone(), one);

                let mut results = vec![next_counter.id];
                for (name, structure) in carried.iter().zip(&init_vals) {
                    let final_val = env2[name].clone();
                    if tval_sig(structure) != tval_sig(&final_val) {
                        die(&format!("loop-carried {} changed shape: {} vs {}",
                                     name, tval_sig(structure), tval_sig(&final_val)));
                    }
                    let mut leaves = Vec::new();
                    collect_leaves(&final_val, &mut leaves);
                    results.extend(leaves.iter().map(|b| b.val.id));
                }

                let body: Vec<usize> = (body_start..self.nodes.len())
                    .filter(|id| !self.claimed.contains(id))
                    .collect();
                self.claimed.extend(body.iter().copied());

                let mut iter_args = vec![counter_arg.id];
                iter_args.extend(arg_leaves.iter().map(|b| b.val.id));
                let mut inputs = vec![counter_init.id];
                inputs.extend(init_leaves_per.iter().flatten().map(|b| b.val.id));

                let w = self.emit(
                    OpKind::While { iter_args, results, body, limit: limit.id },
                    inputs,
                    vec![],
                    Dtype::F64,
                );

                let mut proj_leaves = Vec::new();
                for (k, b) in arg_leaves.iter().enumerate() {
                    let val = self.emit(OpKind::Proj(k + 1), vec![w.id], b.val.shape.clone(), b.val.dtype);
                    proj_leaves.push(val);
                }
                let mut env3 = env.clone();
                let mut proj_iter = proj_leaves.into_iter();
                for (name, structure) in carried.iter().zip(&init_vals) {
                    env3.insert(name.clone(), rebuild(structure, &mut proj_iter));
                }
                self.trace(rest, &env3, fns)
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
                if let Some(callee) = env.get(name) {
                    let callee = callee.clone();
                    let argv: Vec<TVal> = args.iter().map(|a| self.trace(a, env, fns)).collect();
                    self.call_method(callee, "forward", argv, fns)
                } else if self.modules.contains_key(name) {
                    self.instantiate(name, args, env, fns)
                } else if let Some(decl) = fns.get(name) {
                    if decl.params.len() != args.len() {
                        die(&format!("arity mismatch: {} expects {} args, got {}",
                                     name, decl.params.len(), args.len()));
                    }
                    let mut env2 = HashMap::new();
                    for (param, arg) in decl.params.iter().zip(args.iter()) {
                        let v = self.trace(arg, env, fns);
                        env2.insert(param.clone(), v);
                    }
                    self.statics.push(HashMap::new());
                    let out = self.trace(&decl.body, &env2, fns);
                    self.statics.pop();
                    out
                } else {
                    self.builtin(name, args, env, fns)
                }
            }
        }
    }

    fn builtin(&mut self, name: &str, args: &[Expr], env: &HashMap<String, TVal>, fns: &HashMap<String, Decl>) -> TVal {
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
                if self.region_depth > 0 {
                    die("print inside a for loop isn't supported (loops compile to one XLA while op); print after the loop");
                }
                let v = self.trace(&args[0], env, fns);
                if let TVal::Tensor(b) = &v {
                    if b.val.dtype == Dtype::I1 {
                        die("cannot print booleans; use where to select values");
                    }
                }
                self.push_prints(None, &v);
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
                let indices = self.emit(OpKind::Iota, vec![], vec![count as usize], Dtype::F64);
                let step_c = self.constant(step, Dtype::F64);
                let scaled = self.ewise("multiply", indices, step_c);
                let start_c = self.constant(start, Dtype::F64);
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
                let indices = self.emit(OpKind::Iota, vec![], vec![count], Dtype::F64);
                let step_c = self.constant(step, Dtype::F64);
                let scaled = self.ewise("multiply", indices, step_c);
                let start_c = self.constant(start, Dtype::F64);
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
                    let val = self.zeros(&dims, Dtype::F64);
                    return TVal::Tensor(BVal { val, bdims: 0 });
                }
                let count: usize = dims.iter().product();
                let vals: Vec<f64> = (0..count).map(|_| self.next_normal()).collect();
                let val = self.emit(OpKind::DenseConst(vals), vec![], dims, Dtype::F64);
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
            "load" => {
                if args.len() != 1 {
                    die(&format!("load expects 1 arg, got {}", args.len()));
                }
                let path = match &args[0] {
                    Expr::Str(s) => s.clone(),
                    _ => die("load expects a file path string literal"),
                };
                if let Some(&(_, id)) = self.inputs.iter().find(|(p, _)| *p == path) {
                    return TVal::Tensor(BVal { val: self.val(id), bdims: 0 });
                }
                let (shape, dtype, _) = npy_meta(&path);
                let val = self.emit(OpKind::Input, vec![], shape, dtype);
                self.inputs.push((path, val.id));
                TVal::Tensor(BVal { val, bdims: 0 })
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
                    let extras: Vec<TVal> = args[1..].iter().map(|a| self.trace(a, env, fns)).collect();
                    self.grad_depth += 1;
                    let out = self.call_method(inst.clone(), mname, extras, fns);
                    self.grad_depth -= 1;
                    (format!(".{}", mname), inst, out)
                } else {
                    let (fname, decl, vals) = self.transform_args("grad", args, env, fns);
                    let mut env2 = HashMap::new();
                    for (param, v) in decl.params.iter().zip(&vals) {
                        env2.insert(param.clone(), v.clone());
                    }
                    self.statics.push(HashMap::new());
                    self.grad_depth += 1;
                    let out = self.trace(&decl.body, &env2, fns);
                    self.grad_depth -= 1;
                    self.statics.pop();
                    (fname, vals.into_iter().next().unwrap(), out)
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
                let (fname, decl, vals) = self.transform_args("jacobian", args, env, fns);
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

    fn transform_args<'f>(&mut self, transform: &str, args: &[Expr], env: &HashMap<String, TVal>, fns: &'f HashMap<String, Decl>) -> (String, &'f Decl, Vec<TVal>) {
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

fn tval_sig(v: &TVal) -> String {
    match v {
        TVal::Tensor(b) => format!("{:?}:{}", per_shape(b), b.val.dtype.name()),
        TVal::Record(_, fields) => {
            let parts: Vec<String> = fields.iter()
                .map(|(k, f)| format!("{}: {}", k, tval_sig(f)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

fn collect_leaves(v: &TVal, out: &mut Vec<BVal>) {
    match v {
        TVal::Tensor(b) => out.push(b.clone()),
        TVal::Record(_, fields) => {
            for (_, f) in fields {
                collect_leaves(f, out);
            }
        }
    }
}

fn rebuild(structure: &TVal, grads: &mut std::vec::IntoIter<Val>) -> TVal {
    match structure {
        TVal::Tensor(b) => TVal::Tensor(BVal { val: grads.next().unwrap(), bdims: b.bdims }),
        TVal::Record(tag, fields) => TVal::Record(
            tag.clone(),
            fields.iter().map(|(k, f)| (k.clone(), rebuild(f, grads))).collect()
        ),
    }
}

fn named_const(name: &str) -> Option<f64> {
    match name {
        "pi" => Some(std::f64::consts::PI),
        _ => None,
    }
}

impl Tracer {
    fn static_num(&self, name: &str) -> Option<f64> {
        self.statics.last().and_then(|frame| frame.get(name).copied())
    }

    fn num_lit(&self, e: &Expr, env: &HashMap<String, TVal>, what: &str) -> f64 {
        match e {
            Expr::Num(n) => *n,
            Expr::Neg(inner) => -self.num_lit(inner, env, what),
            Expr::Var(s) => {
                if let Some(v) = env.get(s) {
                    if let TVal::Tensor(b) = v {
                        if b.val.shape.is_empty() {
                            if let OpKind::Constant(n) = self.nodes[b.val.id].kind {
                                return n;
                            }
                        }
                    }
                    die(&format!("{} must be a compile-time constant; '{}' is computed at runtime", what, s));
                }
                self.static_num(s)
                    .or_else(|| named_const(s))
                    .unwrap_or_else(|| die(&format!("{} must be a number literal", what)))
            }
            _ => die(&format!("{} must be a number literal", what)),
        }
    }

    fn int_lit(&self, e: &Expr, env: &HashMap<String, TVal>, what: &str) -> usize {
        let n = self.num_lit(e, env, what);
        if n.fract() != 0.0 || n < 0.0 {
            die(&format!("{} must be a non-negative integer literal", what));
        }
        n as usize
    }

    fn axis_lit(&self, e: &Expr, env: &HashMap<String, TVal>, shape: &[usize]) -> usize {
        let n = self.num_lit(e, env, "reduction axis");
        if n.fract() != 0.0 || n < 0.0 || n as usize >= shape.len() {
            die(&format!("reduction axis {} out of range for shape {:?}", n, shape));
        }
        n as usize
    }

    pub fn next_u64(&mut self) -> u64 {
        self.rng = self.rng.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub fn next_normal(&mut self) -> f64 {
        let u1 = ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}
