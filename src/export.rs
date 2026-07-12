use std::collections::{HashMap, HashSet};
use std::fs;

use crate::batch::collect_leaves;
use crate::die;
use crate::emit::build_module;
use crate::graph::{BVal, Dtype, Node, OpKind, TVal, Val};
use crate::parser::Decl;
use crate::runtime::Tensor;
use crate::trace::Tracer;

#[derive(Debug, Clone)]
pub struct ExportSpec {
    pub path: String,
    pub nodes: Vec<Node>,
    pub holes: Vec<usize>,
    pub params: Vec<usize>,
    pub outputs: Vec<Val>,
    pub weight_vals: Vec<Val>,
}

fn clone_with_holes(v: &TVal, sub: &mut Tracer, holes: &mut Vec<usize>) -> TVal {
    match v {
        TVal::Tensor(b) => {
            let val = sub.emit(OpKind::Input, vec![], b.val.shape.clone(), b.val.dtype);
            holes.push(val.id);
            TVal::Tensor(BVal { val, bdims: 0 })
        }
        TVal::Record(tag, fields) => TVal::Record(
            tag.clone(),
            fields.iter().map(|(k, f)| (k.clone(), clone_with_holes(f, sub, holes))).collect(),
        ),
    }
}

impl Tracer {
    pub fn plan_export(&mut self, v: &TVal, path: &str, examples: Vec<TVal>, fns: &HashMap<String, Decl>) {
        if self.region_depth > 0 {
            die("export inside a for loop isn't supported (loops compile to one XLA while op); export after the loop");
        }
        if crate::net::is_url(path) {
            die("cannot export to a url; export locally and upload");
        }
        if !path.ends_with(".mlir") {
            die("export expects a path ending in .mlir");
        }
        if self.exports.iter().any(|e| e.path == path) {
            die(&format!("duplicate export to {}", path));
        }
        if !matches!(v, TVal::Record(Some(_), _)) {
            die("export expects a module instance");
        }
        let mut leaves = Vec::new();
        collect_leaves(v, &mut leaves);
        for b in &leaves {
            if b.val.dtype == Dtype::I1 {
                die("cannot export booleans");
            }
            if b.bdims != 0 {
                die("export inside vmap isn't supported");
            }
        }
        let mut sub = Tracer {
            nodes: Vec::new(),
            prints: Vec::new(),
            inputs: Vec::new(),
            saves: Vec::new(),
            exports: Vec::new(),
            figures: Vec::new(),
            figure: crate::plot::FigureSpec::default(),
            plays: Vec::new(),
            loop_prints: Vec::new(),
            modules: self.modules.clone(),
            statics: Vec::new(),
            rng: self.rng,
            claimed: HashSet::new(),
            region_depth: 0,
            grad_depth: 0,
            interned: vec![HashMap::new()],
        };
        let mut holes = Vec::new();
        let sub_model = clone_with_holes(v, &mut sub, &mut holes);
        let mut ex_tvals = Vec::new();
        for (i, e) in examples.iter().enumerate() {
            let b = match e {
                TVal::Tensor(b) => b,
                TVal::Record(..) => die("export example arguments must be tensors"),
            };
            let val = sub.live_input(format!("arg{}", i), b.val.shape.clone(), b.val.dtype);
            ex_tvals.push(TVal::Tensor(BVal { val, bdims: 0 }));
        }
        let out = sub.call_method(sub_model, "forward", ex_tvals, fns);
        let mut out_leaves = Vec::new();
        collect_leaves(&out, &mut out_leaves);
        self.exports.push(ExportSpec {
            path: path.to_string(),
            holes,
            params: sub.inputs.iter().map(|&(_, id)| id).collect(),
            outputs: out_leaves.iter().map(|b| b.val.clone()).collect(),
            weight_vals: leaves.iter().map(|b| b.val.clone()).collect(),
            nodes: sub.nodes,
        });
    }
}

pub fn write_export(spec: &ExportSpec, tensors: &[Tensor]) {
    let mut nodes = spec.nodes.clone();
    for (&hole, t) in spec.holes.iter().zip(tensors) {
        nodes[hole] = Node {
            kind: OpKind::DenseConst(t.f64_vec()),
            inputs: vec![],
            shape: nodes[hole].shape.clone(),
            dtype: nodes[hole].dtype,
        };
    }
    let text = build_module(&nodes, &spec.params, &spec.outputs);
    fs::write(&spec.path, text)
        .unwrap_or_else(|e| die(&format!("cannot write {}: {}", spec.path, e)));
}
