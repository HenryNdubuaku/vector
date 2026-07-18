use crate::die;
use crate::graph::{broadcast_shape, per_shape, BVal, Dtype, OpKind, TVal, Val};
use crate::parser::Op;
use crate::trace::Tracer;

impl Tracer {
    pub fn binop(&mut self, op: Op, a: TVal, b: TVal) -> TVal {
        let name = match op {
            Op::Add => "add",
            Op::Sub => "subtract",
            Op::Mul => "multiply",
            Op::Div => "divide",
        };
        self.tmap2(name, a, b)
    }

    pub fn tmap2(&mut self, name: &str, a: TVal, b: TVal) -> TVal {
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

    pub fn tunary(&mut self, name: &str, v: &TVal) -> TVal {
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

    pub fn tconvert(&mut self, v: &TVal, dtype: Dtype) -> TVal {
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

    pub fn barrier_tval(&mut self, v: &TVal) -> TVal {
        match v {
            TVal::Tensor(b) => {
                let val = self.emit(OpKind::Barrier, vec![b.val.id], b.val.shape.clone(), b.val.dtype);
                TVal::Tensor(BVal { val, bdims: b.bdims })
            }
            TVal::Record(tag, fields) => {
                let mut out = Vec::new();
                for (k, f) in fields {
                    let r = self.barrier_tval(f);
                    out.push((k.clone(), r));
                }
                TVal::Record(tag.clone(), out)
            }
        }
    }

    pub fn push_prints(&mut self, label: Option<String>, v: &TVal) {
        match v {
            TVal::Tensor(b) => {
                let decode = self.decodes.get(&b.val.id).cloned();
                self.prints.push(crate::trace::PrintSpec { label, val: b.val.clone(), rows: None, decode });
            }
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

    pub fn push_loop_prints(&mut self, label: Option<String>, v: &TVal) {
        match v {
            TVal::Tensor(b) => {
                let decode = self.decodes.get(&b.val.id).cloned();
                self.loop_prints.push((label, b.val.clone(), decode));
            }
            TVal::Record(_, fields) => {
                for (k, f) in fields {
                    let path = match &label {
                        Some(p) => format!("{}.{}", p, k),
                        None => k.clone(),
                    };
                    self.push_loop_prints(Some(path), f);
                }
            }
        }
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

    pub fn align(&mut self, v: &BVal, prefix: &[usize], per: &[usize]) -> Val {
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

    pub fn batch_prefix(a: &BVal, b: &BVal) -> Vec<usize> {
        let deep = if a.bdims >= b.bdims { a } else { b };
        deep.val.shape[..deep.bdims].to_vec()
    }

    pub fn balign(&mut self, a: BVal, b: BVal) -> (Val, Val, Vec<usize>, Vec<usize>) {
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

    pub fn bewise(&mut self, name: &str, a: BVal, b: BVal) -> BVal {
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

    pub fn bcompare(&mut self, dir: &str, a: BVal, b: BVal) -> BVal {
        if a.val.dtype == Dtype::I1 || b.val.dtype == Dtype::I1 {
            die("cannot compare booleans");
        }
        let (av, bv, prefix, per) = self.balign(a, b);
        let mut shape = prefix.clone();
        shape.extend(&per);
        let val = self.emit(OpKind::Compare(dir.to_string()), vec![av.id, bv.id], shape, Dtype::I1);
        BVal { val, bdims: prefix.len() }
    }

    pub fn bunary(&mut self, name: &str, v: &BVal) -> BVal {
        BVal { val: self.unary(name, &v.val), bdims: v.bdims }
    }

    pub fn bmatmul(&mut self, a: BVal, b: BVal) -> BVal {
        if a.val.dtype != b.val.dtype {
            die(&format!("matmul dtype mismatch: {} vs {}", a.val.dtype.name(), b.val.dtype.name()));
        }
        let pa = per_shape(&a);
        let pb = per_shape(&b);
        if pa.is_empty() || pb.is_empty() {
            die(&format!("matmul needs rank >= 1, got {:?} vs {:?}", pa, pb));
        }
        if pa.len() > 2 || pb.len() > 2 {
            return self.batched_matmul(a, b, pa, pb);
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

    fn batched_matmul(&mut self, a: BVal, b: BVal, pa: Vec<usize>, pb: Vec<usize>) -> BVal {
        if pa.len() == 1 || pb.len() == 1 {
            die(&format!("matmul with batch dimensions expects matrices, got {:?} vs {:?}", pa, pb));
        }
        let ba = &pa[..pa.len() - 2];
        let bb = &pb[..pb.len() - 2];
        let per_batch: Vec<usize> = if ba.is_empty() {
            bb.to_vec()
        } else if bb.is_empty() || ba == bb {
            ba.to_vec()
        } else {
            die(&format!("matmul batch dimensions mismatch: {:?} vs {:?}", pa, pb));
        };
        let core_a = &pa[pa.len() - 2..];
        let core_b = &pb[pb.len() - 2..];
        if core_a[1] != core_b[0] {
            die(&format!("matmul contraction mismatch: {:?} vs {:?}", pa, pb));
        }
        let prefix = Self::batch_prefix(&a, &b);
        let k = prefix.len();
        let nb = k + per_batch.len();
        let lift = |t: &mut Tracer, v: &BVal, core: &[usize], has_batch: bool| -> Val {
            let mut target = prefix.clone();
            target.extend(&per_batch);
            target.extend(core);
            let mut dims: Vec<usize> = (0..v.bdims).collect();
            if has_batch {
                dims.extend(k..k + per_shape(v).len());
            } else {
                dims.extend(nb..nb + core.len());
            }
            t.broadcast_along(&v.val, &target, dims)
        };
        let av = lift(self, &a, core_a, pa.len() > 2);
        let bv = lift(self, &b, core_b, pb.len() > 2);
        let batch: Vec<usize> = (0..nb).collect();
        let val = self.dot(&av, &bv, batch.clone(), batch, vec![nb + 1], vec![nb]);
        BVal { val, bdims: a.bdims.max(b.bdims) }
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

    pub fn stack(&mut self, vals: Vec<BVal>) -> BVal {
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
        let mut const_vals: Option<Vec<f64>> = Some(Vec::new());
        for v in &vals {
            if v.bdims != 0 {
                const_vals = None;
                break;
            }
            match &self.nodes[v.val.id].kind {
                OpKind::Constant(n) => {
                    const_vals.as_mut().unwrap().push(*n);
                }
                OpKind::DenseConst(vs) => {
                    const_vals.as_mut().unwrap().extend(vs.iter().copied());
                }
                _ => {
                    const_vals = None;
                    break;
                }
            }
        }
        if let Some(cv) = const_vals {
            let mut shape = vec![vals.len()];
            shape.extend(&inner_shape);
            let val = self.emit(OpKind::DenseConst(cv), vec![], shape, dtype);
            return BVal { val, bdims: 0 };
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

}

pub fn tval_sig(v: &TVal) -> String {
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

pub fn collect_leaves(v: &TVal, out: &mut Vec<BVal>) {
    match v {
        TVal::Tensor(b) => out.push(b.clone()),
        TVal::Record(_, fields) => {
            for (_, f) in fields {
                collect_leaves(f, out);
            }
        }
    }
}

pub fn rebuild(structure: &TVal, grads: &mut std::vec::IntoIter<Val>) -> TVal {
    match structure {
        TVal::Tensor(b) => TVal::Tensor(BVal { val: grads.next().unwrap(), bdims: b.bdims }),
        TVal::Record(tag, fields) => TVal::Record(
            tag.clone(),
            fields.iter().map(|(k, f)| (k.clone(), rebuild(f, grads))).collect()
        ),
    }
}

