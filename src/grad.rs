use std::collections::HashMap;

use crate::die;
use crate::graph::{Dtype, OpKind, Val};
use crate::trace::Tracer;

impl Tracer {
    pub fn backward(&mut self, y: &Val, targets: &[Val], seed: Val) -> Vec<Val> {
        let mut cot: HashMap<usize, Val> = HashMap::new();
        cot.insert(y.id, seed);
        let stop = targets.iter().map(|t| t.id).min().unwrap();
        for id in (stop + 1..=y.id).rev() {
            let Some(g) = cot.get(&id).cloned() else { continue };
            for (input_id, contribution) in self.vjp(id, &g) {
                let merged = match cot.remove(&input_id) {
                    Some(prev) => self.ewise("add", prev, contribution),
                    None => contribution,
                };
                cot.insert(input_id, merged);
            }
        }
        targets.iter()
            .map(|t| match cot.get(&t.id) {
                Some(v) => v.clone(),
                None => self.zeros_like(t),
            })
            .collect()
    }

    fn vjp(&mut self, id: usize, g: &Val) -> Vec<(usize, Val)> {
        let node = self.nodes[id].clone();
        let out = self.val(id);
        let ins: Vec<Val> = node.inputs.iter().map(|&i| self.val(i)).collect();
        match &node.kind {
            OpKind::Input | OpKind::Iota | OpKind::IterArg | OpKind::Constant(_) | OpKind::DenseConst(_) | OpKind::Compare(_) => vec![],
            OpKind::While { .. } | OpKind::Proj(_) => {
                die("differentiating across a for loop isn't supported; take gradients inside the loop body")
            }
            OpKind::Sort { axis, num } => {
                if *num != 1 || *axis != 0 || ins[0].shape.len() != 1 {
                    die("differentiating through sort isn't supported yet");
                }
                let n = ins[0].shape[0];
                let iota = self.emit(OpKind::Iota, vec![], vec![n], Dtype::F32);
                let s1 = self.emit(OpKind::Sort { axis: 0, num: 2 }, vec![ins[0].id, iota.id], vec![n], ins[0].dtype);
                let perm = self.emit(OpKind::Proj(1), vec![s1.id], vec![n], Dtype::F32);
                let s2 = self.emit(OpKind::Sort { axis: 0, num: 2 }, vec![perm.id, iota.id], vec![n], Dtype::F32);
                let inv = self.emit(OpKind::Proj(1), vec![s2.id], vec![n], Dtype::F32);
                let idx = self.convert(&inv, Dtype::I64);
                let dx = self.emit(OpKind::Gather, vec![g.id, idx.id], vec![n], g.dtype);
                vec![(ins[0].id, dx)]
            }
            OpKind::Gather => {
                let zeros = self.zeros_like(&ins[0]);
                let dx = self.emit(OpKind::Scatter, vec![zeros.id, ins[1].id, g.id], ins[0].shape.clone(), ins[0].dtype);
                vec![(ins[0].id, dx)]
            }
            OpKind::Scatter => die("differentiating through scatter isn't supported yet"),
            OpKind::DynSlice(_) => {
                let zeros = self.zeros_like(&ins[0]);
                let mut inputs = vec![zeros.id, g.id];
                inputs.extend(node.inputs[1..].iter().copied());
                let da = self.emit(OpKind::DynUpdateSlice, inputs, ins[0].shape.clone(), ins[0].dtype);
                vec![(ins[0].id, da)]
            }
            OpKind::DynUpdateSlice => {
                die("differentiating through dynamic_update_slice isn't supported yet")
            }
            OpKind::Ewise(name) => match name.as_str() {
                "add" => vec![(ins[0].id, g.clone()), (ins[1].id, g.clone())],
                "subtract" => {
                    let db = self.unary("negate", g);
                    vec![(ins[0].id, g.clone()), (ins[1].id, db)]
                }
                "multiply" => {
                    let da = self.ewise("multiply", g.clone(), ins[1].clone());
                    let db = self.ewise("multiply", g.clone(), ins[0].clone());
                    vec![(ins[0].id, da), (ins[1].id, db)]
                }
                "divide" => {
                    let da = self.ewise("divide", g.clone(), ins[1].clone());
                    let g_out = self.ewise("multiply", g.clone(), out);
                    let quotient = self.ewise("divide", g_out, ins[1].clone());
                    let db = self.unary("negate", &quotient);
                    vec![(ins[0].id, da), (ins[1].id, db)]
                }
                "maximum" | "minimum" => {
                    let dir = if name == "maximum" { "GE" } else { "LE" };
                    let pred = self.compare(dir, &ins[0], &ins[1]);
                    let zero = self.zeros_like(g);
                    let da = self.select(&pred, g, &zero);
                    let db = self.select(&pred, &zero, g);
                    vec![(ins[0].id, da), (ins[1].id, db)]
                }
                _ => die(&format!("no gradient rule for {}", name)),
            },
            OpKind::Unary(name) => {
                let da = match name.as_str() {
                    "floor" => return vec![],
                    "negate" => self.unary("negate", g),
                    "exponential" => self.ewise("multiply", g.clone(), out),
                    "log" => self.ewise("divide", g.clone(), ins[0].clone()),
                    "sqrt" => {
                        let two = self.constant(2.0, node.dtype);
                        let denom = self.ewise("multiply", two, out);
                        self.ewise("divide", g.clone(), denom)
                    }
                    "tanh" => {
                        let one = self.constant(1.0, node.dtype);
                        let squared = self.ewise("multiply", out.clone(), out);
                        let sech2 = self.ewise("subtract", one, squared);
                        self.ewise("multiply", g.clone(), sech2)
                    }
                    "sine" => {
                        let c = self.unary("cosine", &ins[0]);
                        self.ewise("multiply", g.clone(), c)
                    }
                    "cosine" => {
                        let s = self.unary("sine", &ins[0]);
                        let gs = self.ewise("multiply", g.clone(), s);
                        self.unary("negate", &gs)
                    }
                    _ => die(&format!("no gradient rule for {}", name)),
                };
                vec![(ins[0].id, da)]
            }
            OpKind::Convert => vec![(ins[0].id, self.convert(g, ins[0].dtype))],
            OpKind::Broadcast(dims) => {
                let axes: Vec<usize> = (0..node.shape.len()).filter(|d| !dims.contains(d)).collect();
                let da = self.reduce_sum(g, &axes);
                vec![(ins[0].id, da)]
            }
            OpKind::Reshape => vec![(ins[0].id, self.reshape(g, ins[0].shape.clone()))],
            OpKind::Transpose(perm) => {
                let mut inv = vec![0; perm.len()];
                for (i, &p) in perm.iter().enumerate() {
                    inv[p] = i;
                }
                let da = self.emit(OpKind::Transpose(inv), vec![g.id], ins[0].shape.clone(), g.dtype);
                vec![(ins[0].id, da)]
            }
            OpKind::Barrier => vec![(ins[0].id, g.clone())],
            OpKind::Concat(dim) => {
                let mut contribs = Vec::new();
                let mut offset = 0;
                for &input_id in &node.inputs {
                    let shape = self.nodes[input_id].shape.clone();
                    let extent = shape[*dim];
                    let piece = self.emit(OpKind::Slice(*dim, offset, offset + extent), vec![g.id], shape, g.dtype);
                    contribs.push((input_id, piece));
                    offset += extent;
                }
                contribs
            }
            OpKind::Reduce(reducer, axes) => {
                let kept: Vec<usize> = (0..ins[0].shape.len()).filter(|d| !axes.contains(d)).collect();
                let gb = self.broadcast_along(g, &ins[0].shape.clone(), kept.clone());
                let da = match reducer.as_str() {
                    "add" => gb,
                    _ => {
                        let ob = self.broadcast_along(&out, &ins[0].shape.clone(), kept);
                        let mask = self.compare("EQ", &ins[0], &ob);
                        let zero = self.zeros_like(&gb);
                        self.select(&mask, &gb, &zero)
                    }
                };
                vec![(ins[0].id, da)]
            }
            OpKind::Dot(lb, _, lc, rc) => {
                let (a, b) = (ins[0].clone(), ins[1].clone());
                let k = lb.len();
                if *lc != vec![a.shape.len() - 1] || *rc != vec![k] {
                    die("higher-order gradients through matmul aren't supported yet");
                }
                let batch: Vec<usize> = (0..k).collect();
                let (da, db) = match (a.shape.len() - k, b.shape.len() - k) {
                    (2, 2) => (
                        self.dot(g, &b, batch.clone(), batch.clone(), vec![k + 1], vec![k + 1]),
                        self.dot(&a, g, batch.clone(), batch.clone(), vec![k], vec![k]),
                    ),
                    (1, 2) => (
                        self.dot(&b, g, batch.clone(), batch.clone(), vec![k + 1], vec![k]),
                        self.dot(&a, g, batch.clone(), batch, vec![], vec![]),
                    ),
                    (2, 1) => (
                        self.dot(g, &b, batch.clone(), batch.clone(), vec![], vec![]),
                        self.dot(&a, g, batch.clone(), batch, vec![k], vec![k]),
                    ),
                    _ => {
                        let gb = if k == 0 { g.clone() } else { self.broadcast_along(g, &b.shape.clone(), batch.clone()) };
                        let ga = if k == 0 { g.clone() } else { self.broadcast_along(g, &a.shape.clone(), batch) };
                        (self.ewise("multiply", gb, b.clone()), self.ewise("multiply", ga, a.clone()))
                    }
                };
                vec![(a.id, da), (b.id, db)]
            }
            OpKind::Select => {
                let zero = self.zeros_like(g);
                let dt = self.select(&ins[0], g, &zero);
                let df = self.select(&ins[0], &zero, g);
                vec![(ins[1].id, dt), (ins[2].id, df)]
            }
            OpKind::Slice(dim, start, limit) => {
                let in_shape = ins[0].shape.clone();
                let mut parts = Vec::new();
                if *start > 0 {
                    let mut shape = in_shape.clone();
                    shape[*dim] = *start;
                    parts.push(self.zeros(&shape, g.dtype).id);
                }
                parts.push(g.id);
                if *limit < in_shape[*dim] {
                    let mut shape = in_shape.clone();
                    shape[*dim] = in_shape[*dim] - *limit;
                    parts.push(self.zeros(&shape, g.dtype).id);
                }
                let da = if parts.len() == 1 {
                    g.clone()
                } else {
                    self.emit(OpKind::Concat(*dim), parts, in_shape, g.dtype)
                };
                vec![(ins[0].id, da)]
            }
        }
    }
}
