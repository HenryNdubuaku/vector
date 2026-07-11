use crate::die;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dtype { F32, F64, I1 }

impl Dtype {
    pub fn name(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F64 => "f64",
            Dtype::I1 => "i1",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Val {
    pub id: usize,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

#[derive(Debug, Clone)]
pub struct BVal {
    pub val: Val,
    pub bdims: usize,
}

pub fn per_shape(v: &BVal) -> Vec<usize> {
    v.val.shape[v.bdims..].to_vec()
}

pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    if a == b {
        a.to_vec()
    } else if a.len() <= b.len() && b.ends_with(a) {
        b.to_vec()
    } else if b.len() < a.len() && a.ends_with(b) {
        a.to_vec()
    } else {
        die(&format!("shape mismatch: {:?} vs {:?} (broadcast aligns trailing dims)", a, b));
    }
}

#[derive(Debug, Clone)]
pub enum OpKind {
    Input,
    Constant(f64),
    Ewise(String),
    Unary(String),
    Convert,
    Broadcast(Vec<usize>),
    Reshape,
    Reduce(Vec<usize>),
    Dot(Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>),
    Compare(String),
    Select,
    Concat(usize),
    Slice(usize, usize, usize),
}

#[derive(Debug, Clone)]
pub struct Node {
    pub kind: OpKind,
    pub inputs: Vec<usize>,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}
