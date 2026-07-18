use crate::die;
use crate::graph::{per_shape, BVal, Dtype, OpKind, TVal, Val};
use crate::trace::Tracer;

fn vector_only(v: &BVal, what: &str) -> usize {
    let per = per_shape(v);
    if per.len() != 1 {
        die(&format!("{} expects a vector, got shape {:?} (use vmap for batches)", what, per));
    }
    per[0]
}

impl Tracer {
    fn iota_along(&mut self, shape: &[usize], dim: usize, dtype: Dtype) -> Val {
        let idx = self.emit(OpKind::Iota, vec![], vec![shape[dim]], Dtype::F32);
        let idx = self.convert(&idx, dtype);
        if shape.len() == 1 {
            return idx;
        }
        self.broadcast_along(&idx, shape, vec![dim])
    }

    pub fn sort_val(&mut self, v: BVal) -> TVal {
        vector_only(&v, "sort");
        let val = self.emit(
            OpKind::Sort { axis: v.bdims, num: 1 },
            vec![v.val.id],
            v.val.shape.clone(),
            v.val.dtype,
        );
        TVal::Tensor(BVal { val, bdims: v.bdims })
    }

    pub fn argsort_val(&mut self, v: BVal) -> TVal {
        vector_only(&v, "argsort");
        let shape = v.val.shape.clone();
        let idx = self.iota_along(&shape, v.bdims, Dtype::F32);
        let s = self.emit(
            OpKind::Sort { axis: v.bdims, num: 2 },
            vec![v.val.id, idx.id],
            shape.clone(),
            v.val.dtype,
        );
        let val = self.emit(OpKind::Proj(1), vec![s.id], shape, Dtype::F32);
        TVal::Tensor(BVal { val, bdims: v.bdims })
    }

    pub fn arg_extreme(&mut self, v: BVal, largest: bool) -> TVal {
        let what = if largest { "argmax" } else { "argmin" };
        let per = per_shape(&v);
        if per.is_empty() {
            die(&format!("{} expects rank >= 1, got a scalar", what));
        }
        let n: usize = per.iter().product();
        let mut fshape: Vec<usize> = v.val.shape[..v.bdims].to_vec();
        fshape.push(n);
        let flat = self.reshape(&v.val, fshape.clone());
        let (reducer, init) = if largest {
            ("maximum", f64::NEG_INFINITY)
        } else {
            ("minimum", f64::INFINITY)
        };
        let ext = self.reduce(reducer, init, &flat, &[v.bdims]);
        let ext_b = self.broadcast_along(&ext, &fshape, (0..v.bdims).collect());
        let mask = self.emit(OpKind::Compare("EQ".to_string()), vec![flat.id, ext_b.id], fshape.clone(), Dtype::I1);
        let idx = self.iota_along(&fshape, v.bdims, v.val.dtype);
        let big = self.constant(n as f64, v.val.dtype);
        let big_b = self.broadcast(&big, &fshape);
        let masked = self.select(&mask, &idx, &big_b);
        let arg = self.reduce("minimum", f64::INFINITY, &masked, &[v.bdims]);
        let arg = self.convert(&arg, Dtype::F32);
        TVal::Tensor(BVal { val: arg, bdims: v.bdims })
    }

    pub fn one_hot_val(&mut self, v: BVal, n: usize) -> TVal {
        let per = per_shape(&v);
        if per.len() > 1 {
            die(&format!("one_hot expects a scalar or vector, got shape {:?}", per));
        }
        if n == 0 {
            die("one_hot depth must be positive");
        }
        let vi = self.convert(&v.val, Dtype::F32);
        let mut shape = v.val.shape.clone();
        shape.push(n);
        let vb = self.broadcast_along(&vi, &shape, (0..vi.shape.len()).collect());
        let ib = self.iota_along(&shape, shape.len() - 1, Dtype::F32);
        let mask = self.emit(OpKind::Compare("EQ".to_string()), vec![vb.id, ib.id], shape.clone(), Dtype::I1);
        let one = self.constant(1.0, Dtype::F32);
        let one_b = self.broadcast(&one, &shape);
        let zero_b = self.zeros(&shape, Dtype::F32);
        let val = self.select(&mask, &one_b, &zero_b);
        TVal::Tensor(BVal { val, bdims: v.bdims })
    }

    pub fn take_val(&mut self, x: BVal, idx: BVal) -> TVal {
        let xper = per_shape(&x);
        if xper.is_empty() || xper.len() > 2 {
            die(&format!("take expects rank 1 or 2, got shape {:?}", xper));
        }
        if x.bdims > 0 || idx.bdims > 0 {
            let TVal::Tensor(oh) = self.one_hot_val(idx, xper[0]) else { unreachable!() };
            let ohv = self.convert(&oh.val, x.val.dtype);
            let oh = BVal { val: ohv, bdims: oh.bdims };
            return TVal::Tensor(self.bmatmul(oh, x));
        }
        let iper = per_shape(&idx);
        if iper.len() > 1 {
            die(&format!("take indices must be a scalar or vector, got shape {:?}", iper));
        }
        let scalar = iper.is_empty();
        let iv = if scalar { self.reshape(&idx.val, vec![1]) } else { idx.val.clone() };
        let iv = self.convert(&iv, Dtype::I64);
        let mut shape = vec![iv.shape[0]];
        shape.extend(&x.val.shape[1..]);
        let val = self.emit(OpKind::Gather(1), vec![x.val.id, iv.id], shape, x.val.dtype);
        let val = if scalar { self.reshape(&val, x.val.shape[1..].to_vec()) } else { val };
        TVal::Tensor(BVal { val, bdims: 0 })
    }

    pub fn cumsum_val(&mut self, v: BVal) -> TVal {
        let n = vector_only(&v, "cumsum");
        let ri = self.iota_along(&[n, n], 0, v.val.dtype);
        let ci = self.iota_along(&[n, n], 1, v.val.dtype);
        let mask = self.emit(OpKind::Compare("LE".to_string()), vec![ci.id, ri.id], vec![n, n], Dtype::I1);
        let one = self.constant(1.0, v.val.dtype);
        let one_b = self.broadcast(&one, &[n, n]);
        let zero_b = self.zeros(&[n, n], v.val.dtype);
        let tri = self.select(&mask, &one_b, &zero_b);
        TVal::Tensor(self.bmatmul(BVal { val: tri, bdims: 0 }, v))
    }
}
