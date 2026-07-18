use std::collections::HashMap;

use crate::die;
use crate::graph::{BVal, Dtype, OpKind, TVal};
use crate::lexer::lex;
use crate::parser::{ModuleDecl, Parser};
use crate::trace::Tracer;

pub const SOURCE: &str = "
module Linear(in_size, out_size):
  w = glorot_uniform(in_size, out_size)
  b = zeros(out_size)

  forward(self, x):
    matmul(x, self.w) + self.b

module LayerNorm(dim):
  gain = zeros(dim) + 1.0
  bias = zeros(dim)

  forward(self, x):
    vmap(token_norm, {gain: self.gain, bias: self.bias}, x)

module Embedding(count, dim):
  w = randn(count, dim) * 0.02

  forward(self, ids):
    take(self.w, ids)

function token_norm(params, token):
  layer_norm(token, params.gain, params.bias)

function softmax_rows(rows):
  vmap(softmax, rows)

function relu(x):
  maximum(x, 0.0)

function sigmoid(x):
  1.0 / (1.0 + exp(-x))

function logsumexp(x):
  m = max(x)
  m + log(sum(exp(x - m)))

function softmax(x):
  e = exp(x - max(x))
  e / sum(e)

function var(x):
  d = x - mean(x)
  mean(d * d)

function std(x):
  sqrt(var(x))

function norm(x):
  sqrt(sum(x * x))

function permutation(n):
  argsort(uniform(n))

function random_windows(data, count, size):
  slice(data, floor(uniform(count) * (len(data) - size + 1.0)), size)

function layer_norm(x, g, b):
  (x - mean(x)) / sqrt(var(x) + 0.00001) * g + b

function mse(pred, target):
  d = pred - target
  mean(d * d)

function cross_entropy(logits, target):
  logsumexp(logits) - logits[target]

function clip(x, lo, hi):
  minimum(maximum(x, lo), hi)

function clip_by_norm(x, max_norm):
  x * minimum(1.0, max_norm / norm(x))

function cosine_decay(lr, step, total):
  lr * 0.5 * (1.0 + cos(pi * step / total))

function warmup(lr, step, steps):
  lr * minimum(1.0, step / steps)

function adam_init(params):
  {p: params, m: params * 0.0, v: params * 0.0, k: 0.0}

function adam(st, g, lr):
  m = 0.9 * st.m + 0.1 * g
  v = 0.999 * st.v + 0.001 * g * g
  k = st.k + 1.0
  mh = m / (1.0 - pow(0.9, k))
  vh = v / (1.0 - pow(0.999, k))
  {p: st.p - lr * mh / (sqrt(vh) + 0.00000001), m: m, v: v, k: k}

function adamw(st, g, lr, decay):
  m = 0.9 * st.m + 0.1 * g
  v = 0.999 * st.v + 0.001 * g * g
  k = st.k + 1.0
  mh = m / (1.0 - pow(0.9, k))
  vh = v / (1.0 - pow(0.999, k))
  {p: st.p - lr * (mh / (sqrt(vh) + 0.00000001) + decay * st.p), m: m, v: v, k: k}

function sgd_init(params):
  {p: params, m: params * 0.0}

function sgd(st, g, lr, momentum):
  m = momentum * st.m + g
  {p: st.p - lr * m, m: m}
";

pub fn stdlib() -> (HashMap<String, crate::parser::Decl>, HashMap<String, ModuleDecl>) {
    let src = format!("{}\n0.0\n", SOURCE);
    let lexed = lex(&src);
    let mut p = Parser {
        repl: false,
        library: false,
        toks: lexed.toks,
        cols: lexed.cols,
        lines: lexed.lines,
        pos: 0,
        imports: Vec::new(),
        fns: HashMap::new(),
        modules: HashMap::new(),
    };
    let prog = p.program();
    (prog.fns, prog.modules)
}

impl Tracer {
    fn next_uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    pub fn initializer(&mut self, name: &str, fan_in: usize, fan_out: usize) -> TVal {
        if fan_in == 0 || fan_out == 0 {
            die(&format!("{} needs positive fan_in and fan_out", name));
        }
        let (fi, fo) = (fan_in as f64, fan_out as f64);
        let count = fan_in * fan_out;
        let vals: Vec<f64> = match name {
            "glorot_uniform" => {
                let bound = (6.0 / (fi + fo)).sqrt();
                (0..count).map(|_| self.next_uniform() * bound).collect()
            }
            "glorot_normal" => {
                let std = (2.0 / (fi + fo)).sqrt();
                (0..count).map(|_| self.next_normal() * std).collect()
            }
            "he_uniform" => {
                let bound = (6.0 / fi).sqrt();
                (0..count).map(|_| self.next_uniform() * bound).collect()
            }
            "he_normal" => {
                let std = (2.0 / fi).sqrt();
                (0..count).map(|_| self.next_normal() * std).collect()
            }
            "lecun_uniform" => {
                let bound = (3.0 / fi).sqrt();
                (0..count).map(|_| self.next_uniform() * bound).collect()
            }
            "lecun_normal" => {
                let std = (1.0 / fi).sqrt();
                (0..count).map(|_| self.next_normal() * std).collect()
            }
            _ => die(&format!("unknown initializer: {}", name)),
        };
        let val = self.emit(OpKind::DenseConst(vals), vec![], vec![fan_in, fan_out], Dtype::F32);
        TVal::Tensor(BVal { val, bdims: 0 })
    }
}
