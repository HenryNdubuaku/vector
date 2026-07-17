use std::collections::{HashMap, HashSet};

use crate::batch::{collect_leaves, rebuild, tval_sig};
use crate::die;
use crate::export::ExportSpec;
use crate::graph::{BVal, Dtype, InputSource, Node, OpKind, TVal, Val};
use crate::parser::{Decl, Expr, ModuleDecl};
use crate::plot::FigureSpec;
use crate::safetensors::SaveSpec;

#[derive(Debug, Clone)]
pub struct RowMeta {
    pub var: String,
    pub start: f64,
    pub step: f64,
}

#[derive(Debug, Clone)]
pub struct PrintSpec {
    pub label: Option<String>,
    pub val: Val,
    pub rows: Option<RowMeta>,
}

pub struct Tracer {
    pub nodes: Vec<Node>,
    pub prints: Vec<PrintSpec>,
    pub loop_prints: Vec<(Option<String>, Val)>,
    pub inputs: Vec<(InputSource, usize)>,
    pub saves: Vec<SaveSpec>,
    pub exports: Vec<ExportSpec>,
    pub figures: Vec<FigureSpec>,
    pub figure: FigureSpec,
    pub plays: Vec<SaveSpec>,
    pub modules: HashMap<String, ModuleDecl>,
    pub statics: Vec<HashMap<String, f64>>,
    pub rng: u64,
    pub rng_sites: usize,
    pub rng_baked: bool,
    pub seed: Option<Val>,
    pub loop_counters: Vec<(Val, usize)>,
    pub claimed: HashSet<usize>,
    pub region_depth: usize,
    pub grad_depth: usize,
    pub interned: Vec<HashMap<String, usize>>,
}

impl Tracer {
    pub fn emit(&mut self, kind: OpKind, inputs: Vec<usize>, shape: Vec<usize>, dtype: Dtype) -> Val {
        let internable = !matches!(kind, OpKind::Input | OpKind::IterArg | OpKind::Proj(_) | OpKind::Barrier | OpKind::While { .. } | OpKind::Sort { .. } | OpKind::RngBits);
        if internable {
            let key = format!("{:?}|{:?}|{:?}|{:?}", kind, inputs, shape, dtype);
            for frame in self.interned.iter().rev() {
                if let Some(&id) = frame.get(&key) {
                    return self.val(id);
                }
            }
            let id = self.nodes.len();
            self.nodes.push(Node { kind, inputs, shape: shape.clone(), dtype });
            self.interned.last_mut().unwrap().insert(key, id);
            return Val { id, shape, dtype };
        }
        let id = self.nodes.len();
        self.nodes.push(Node { kind, inputs, shape: shape.clone(), dtype });
        Val { id, shape, dtype }
    }

    pub fn live_input(&mut self, key: String, shape: Vec<usize>, dtype: Dtype) -> Val {
        let val = self.emit(OpKind::Input, vec![], shape, dtype);
        self.inputs.push((InputSource::Live(key), val.id));
        val
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

    pub fn trace(&mut self, e: &Expr, env: &HashMap<String, TVal>, fns: &HashMap<String, Decl>) -> TVal {
        match e {
            Expr::Num(n) => {
                let val = self.constant(*n, Dtype::F32);
                TVal::Tensor(BVal { val, bdims: 0 })
            }
            Expr::Unit => die("internal: unit expression traced"),
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
                    let val = self.constant(n, Dtype::F32);
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
            Expr::For(var, start_e, end_e, step_e, stmts, rest) => {
                let env3 = self.trace_for(var, start_e, end_e, step_e, stmts, env, fns);
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

    pub fn trace_for(&mut self, var: &String, start_e: &Expr, end_e: &Expr, step_e: &Option<Box<Expr>>, stmts: &[(Option<String>, Expr)], env: &HashMap<String, TVal>, fns: &HashMap<String, Decl>) -> HashMap<String, TVal> {

                let start = self.num_lit(start_e, env, "range start");
                let end = self.num_lit(end_e, env, "range end");
                let step = match step_e {
                    Some(e) => self.num_lit(e, env, "range step"),
                    None => 1.0,
                };
                if step == 0.0 {
                    die("range step must be nonzero");
                }
                let iterations = (((end - start) / step).ceil().max(0.0)) as usize;
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
                    let outer_prints = std::mem::take(&mut self.loop_prints);
                    self.region_depth += 1;
                    for k in 0..iterations {
                        let kv = self.constant(start + k as f64 * step, Dtype::F32);
                        env2.insert(var.clone(), TVal::Tensor(BVal { val: kv, bdims: 0 }));
                        for (name, stmt) in stmts {
                            let v = self.trace(stmt, &env2, fns);
                            if let Some(name) = name {
                                env2.insert(name.clone(), v);
                            }
                        }
                        let drained = std::mem::take(&mut self.loop_prints);
                        if !drained.is_empty() && self.region_depth > 1 {
                            die("print inside nested loops isn't supported; print in the outer loop");
                        }
                        for (label, val) in drained {
                            let tag = format!("{} {}", var, start + k as f64 * step);
                            let label = match label {
                                Some(l) => format!("{}: {}", tag, l),
                                None => tag,
                            };
                            self.prints.push(PrintSpec { label: Some(label), val, rows: None });
                        }
                    }
                    self.region_depth -= 1;
                    self.loop_prints = outer_prints;
                    let mut env3 = env.clone();
                    for name in &carried {
                        env3.insert(name.clone(), env2[name].clone());
                    }
                    return env3;
                }
                let init_vals: Vec<TVal> = carried.iter().map(|n| env[n].clone()).collect();
                let init_leaves_per: Vec<Vec<BVal>> = init_vals.iter().map(|v| {
                    let mut l = Vec::new();
                    collect_leaves(v, &mut l);
                    l
                }).collect();

                let limit = self.constant(iterations as f64, Dtype::I64);
                let counter_init = self.constant(0.0, Dtype::I64);

                let body_start = self.nodes.len();
                self.interned.push(HashMap::new());
                let counter_arg = self.emit(OpKind::IterArg, vec![], vec![], Dtype::I64);
                let arg_leaves: Vec<BVal> = init_leaves_per.iter().flatten().map(|b| {
                    let val = self.emit(OpKind::IterArg, vec![], b.val.shape.clone(), b.val.dtype);
                    BVal { val, bdims: b.bdims }
                }).collect();

                let mut env2 = env.clone();
                let mut var_view = self.convert(&counter_arg, Dtype::F32);
                if step != 1.0 {
                    let step_c = self.constant(step, Dtype::F32);
                    var_view = self.ewise("multiply", var_view, step_c);
                }
                if start != 0.0 {
                    let start_c = self.constant(start, Dtype::F32);
                    var_view = self.ewise("add", var_view, start_c);
                }
                env2.insert(var.clone(), TVal::Tensor(BVal { val: var_view, bdims: 0 }));
                let mut arg_iter = arg_leaves.iter().map(|b| b.val.clone()).collect::<Vec<_>>().into_iter();
                for (name, structure) in carried.iter().zip(&init_vals) {
                    env2.insert(name.clone(), rebuild(structure, &mut arg_iter));
                }

                let outer_prints = std::mem::take(&mut self.loop_prints);
                self.region_depth += 1;
                self.loop_counters.push((counter_arg.clone(), iterations));
                for (name, stmt) in stmts {
                    let v = self.trace(stmt, &env2, fns);
                    if let Some(name) = name {
                        env2.insert(name.clone(), v);
                    }
                }
                self.loop_counters.pop();
                self.region_depth -= 1;
                let my_prints = std::mem::take(&mut self.loop_prints);
                if !my_prints.is_empty() && self.region_depth > 0 {
                    die("print inside nested loops isn't supported; print in the outer loop");
                }
                self.loop_prints = outer_prints;

                let one = self.constant(1.0, Dtype::I64);
                let next_counter = self.ewise("add", counter_arg.clone(), one);

                let mut print_bufs = Vec::new();
                for (label, val) in &my_prints {
                    let mut bshape = vec![iterations];
                    bshape.extend(&val.shape);
                    let buf_arg = self.emit(OpKind::IterArg, vec![], bshape.clone(), val.dtype);
                    let mut ushape = vec![1];
                    ushape.extend(&val.shape);
                    let upd = self.reshape(val, ushape);
                    let zero = self.constant(0.0, Dtype::I64);
                    let mut dus_inputs = vec![buf_arg.id, upd.id, counter_arg.id];
                    dus_inputs.extend(std::iter::repeat(zero.id).take(val.shape.len()));
                    let updated = self.emit(OpKind::DynUpdateSlice, dus_inputs, bshape.clone(), val.dtype);
                    print_bufs.push((label.clone(), buf_arg, updated, bshape));
                }

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
                results.extend(print_bufs.iter().map(|(_, _, updated, _)| updated.id));

                let body: Vec<usize> = (body_start..self.nodes.len())
                    .filter(|id| !self.claimed.contains(id))
                    .collect();
                self.claimed.extend(body.iter().copied());
                self.interned.pop();

                let mut iter_args = vec![counter_arg.id];
                iter_args.extend(arg_leaves.iter().map(|b| b.val.id));
                iter_args.extend(print_bufs.iter().map(|(_, arg, _, _)| arg.id));
                let mut inputs = vec![counter_init.id];
                inputs.extend(init_leaves_per.iter().flatten().map(|b| b.val.id));
                for (_, arg, _, bshape) in &print_bufs {
                    inputs.push(self.zeros(bshape, arg.dtype).id);
                }

                let w = self.emit(
                    OpKind::While { iter_args, results, body, limit: limit.id },
                    inputs,
                    vec![],
                    Dtype::I64,
                );

                let mut proj_leaves = Vec::new();
                for (k, b) in arg_leaves.iter().enumerate() {
                    let val = self.emit(OpKind::Proj(k + 1), vec![w.id], b.val.shape.clone(), b.val.dtype);
                    proj_leaves.push(val);
                }
                for (j, (label, arg, _, bshape)) in print_bufs.iter().enumerate() {
                    let val = self.emit(OpKind::Proj(1 + arg_leaves.len() + j), vec![w.id], bshape.clone(), arg.dtype);
                    self.prints.push(PrintSpec {
                        label: label.clone(),
                        val,
                        rows: Some(RowMeta { var: var.clone(), start, step }),
                    });
                }
                let mut env3 = env.clone();
                let mut proj_iter = proj_leaves.into_iter();
                for (name, structure) in carried.iter().zip(&init_vals) {
                    env3.insert(name.clone(), rebuild(structure, &mut proj_iter));
                }
                env3
    }


}

fn named_const(name: &str) -> Option<f64> {
    match name {
        "pi" => Some(std::f64::consts::PI),
        _ => None,
    }
}

impl Tracer {
    pub fn static_num(&self, name: &str) -> Option<f64> {
        self.statics.last().and_then(|frame| frame.get(name).copied())
    }

    pub fn num_lit(&self, e: &Expr, env: &HashMap<String, TVal>, what: &str) -> f64 {
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

    pub fn int_lit(&self, e: &Expr, env: &HashMap<String, TVal>, what: &str) -> usize {
        let n = self.num_lit(e, env, what);
        if n.fract() != 0.0 || n < 0.0 {
            die(&format!("{} must be a non-negative integer literal", what));
        }
        n as usize
    }

    pub fn axis_lit(&self, e: &Expr, env: &HashMap<String, TVal>, shape: &[usize]) -> usize {
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

    fn seed_val(&mut self) -> Val {
        if let Some(seed) = &self.seed {
            return seed.clone();
        }
        let seed = if self.rng_baked {
            let n = (self.next_u64() >> 40) as f64;
            self.constant(n, Dtype::F32)
        } else {
            let val = self.emit(OpKind::Input, vec![], vec![], Dtype::F32);
            self.inputs.push((InputSource::Seed, val.id));
            val
        };
        self.seed = Some(seed.clone());
        seed
    }

    pub fn rng_uniform(&mut self, shape: &[usize]) -> Val {
        let seed = self.seed_val();
        self.rng_sites += 1;
        let site = self.constant(self.rng_sites as f64, Dtype::I64);
        let seed_i = self.convert(&seed, Dtype::I64);
        let s0 = self.ewise("add", seed_i, site);
        let mut ctr = self.constant(0.0, Dtype::I64);
        for (counter, trips) in self.loop_counters.clone() {
            let radix = self.constant(trips as f64, Dtype::I64);
            let scaled = self.ewise("multiply", ctr, radix);
            ctr = self.ewise("add", scaled, counter);
        }
        let s0 = self.convert(&s0, Dtype::U64);
        let s1 = self.convert(&ctr, Dtype::U64);
        let s0 = self.reshape(&s0, vec![1]);
        let s1 = self.reshape(&s1, vec![1]);
        let state = self.emit(OpKind::Concat(0), vec![s0.id, s1.id], vec![2], Dtype::U64);
        let bits_shape = if shape.is_empty() { vec![1] } else { shape.to_vec() };
        let rng = self.emit(OpKind::RngBits, vec![state.id], bits_shape.clone(), Dtype::I32);
        let bits = self.emit(OpKind::Proj(1), vec![rng.id], bits_shape.clone(), Dtype::I32);
        let f = self.convert(&bits, Dtype::F32);
        let scale = self.constant(1.0 / 4294967296.0, Dtype::F32);
        let scaled = self.ewise("multiply", f, scale);
        let whole = self.unary("floor", &scaled);
        let u = self.ewise("subtract", scaled, whole);
        if shape.is_empty() { self.reshape(&u, vec![]) } else { u }
    }
}
