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
";

pub fn stdlib_modules() -> HashMap<String, ModuleDecl> {
    let src = format!("{}\n0.0\n", SOURCE);
    let lexed = lex(&src);
    let mut p = Parser {
        toks: lexed.toks,
        cols: lexed.cols,
        lines: lexed.lines,
        pos: 0,
        fns: HashMap::new(),
        modules: HashMap::new(),
    };
    p.program().modules
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
