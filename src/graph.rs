use crate::die;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dtype { F32, F64, I1, I32, I64, U32 }

impl Dtype {
    pub fn name(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F64 => "f64",
            Dtype::I1 => "i1",
            Dtype::I32 => "i32",
            Dtype::I64 => "i64",
            Dtype::U32 => "ui32",
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

#[derive(Debug, Clone)]
pub enum InputSource {
    Npy(String),
    Safetensors(String, String),
    Csv(String, String),
    Image(String),
    Audio(String),
    Text(String),
    Tokens(String, String),
    Seed,
    Live(String),
}

#[derive(Debug, Clone)]
pub struct ModTag {
    pub module: String,
    pub statics: Vec<(String, f64)>,
}

#[derive(Debug, Clone)]
pub enum TVal {
    Tensor(BVal),
    Record(Option<ModTag>, Vec<(String, TVal)>),
}

impl TVal {
    pub fn tensor(self, what: &str) -> BVal {
        match self {
            TVal::Tensor(b) => b,
            TVal::Record(..) => die(&format!("{} cannot be a record", what)),
        }
    }
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
    Iota,
    Constant(f64),
    DenseConst(Vec<f64>),
    Ewise(String),
    Unary(String),
    Convert,
    Broadcast(Vec<usize>),
    Reshape,
    Reduce(String, Vec<usize>),
    Transpose(Vec<usize>),
    Dot(Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>),
    Compare(String),
    Select,
    Sort { axis: usize, num: usize },
    Gather(usize),
    Scatter,
    Concat(usize),
    Conv {
        stride: usize,
        pad_lo: i64,
        pad_hi: i64,
        lhs_dilation: usize,
        rhs_dilation: usize,
    },
    Reverse(Vec<usize>),
    Slice(usize, usize, usize),
    DynUpdateSlice,
    IterArg,
    Proj(usize),
    Barrier,
    While {
        iter_args: Vec<usize>,
        results: Vec<usize>,
        body: Vec<usize>,
        limit: usize,
        cond: Option<(Vec<usize>, usize)>,
    },
}

#[derive(Debug, Clone)]
pub struct Node {
    pub kind: OpKind,
    pub inputs: Vec<usize>,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}
