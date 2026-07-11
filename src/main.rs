use std::collections::HashMap;
use std::env;
use std::fs;
use std::process::{exit, Command};

use pjrt::ProgramFormat::MLIR;
use pjrt::{Client, HostBuffer, LoadedExecutable};

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f64),
    Ident(String),
    Fn,
    Eq,
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Colon,
}

struct Lexed {
    toks: Vec<Tok>,
    cols: Vec<usize>,
    lines: Vec<usize>,
}

fn lex(src: &str) -> Lexed {
    let mut toks = Vec::new();
    let mut cols = Vec::new();
    let mut lines = Vec::new();
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut line = 1usize;
    let mut col = 1usize;
    let push = |tok: Tok, l: usize, c: usize, toks: &mut Vec<Tok>, lines: &mut Vec<usize>, cols: &mut Vec<usize>| {
        toks.push(tok);
        lines.push(l);
        cols.push(c);
    };
    while i < chars.len() {
        let tl = line;
        let tc = col;
        let c = chars[i];
        match c {
            ' ' | '\t' => { i += 1; col += 1; }
            '\r' => { i += 1; }
            '\n' => { i += 1; line += 1; col = 1; }
            '-' => {
                i += 1; col += 1;
                if i < chars.len() && chars[i] == '-' {
                    while i < chars.len() && chars[i] != '\n' {
                        i += 1; col += 1;
                    }
                } else {
                    push(Tok::Minus, tl, tc, &mut toks, &mut lines, &mut cols);
                }
            }
            '+' => { i += 1; col += 1; push(Tok::Plus, tl, tc, &mut toks, &mut lines, &mut cols); }
            '*' => { i += 1; col += 1; push(Tok::Star, tl, tc, &mut toks, &mut lines, &mut cols); }
            '/' => { i += 1; col += 1; push(Tok::Slash, tl, tc, &mut toks, &mut lines, &mut cols); }
            '(' => { i += 1; col += 1; push(Tok::LParen, tl, tc, &mut toks, &mut lines, &mut cols); }
            ')' => { i += 1; col += 1; push(Tok::RParen, tl, tc, &mut toks, &mut lines, &mut cols); }
            '[' => { i += 1; col += 1; push(Tok::LBracket, tl, tc, &mut toks, &mut lines, &mut cols); }
            ']' => { i += 1; col += 1; push(Tok::RBracket, tl, tc, &mut toks, &mut lines, &mut cols); }
            ',' => { i += 1; col += 1; push(Tok::Comma, tl, tc, &mut toks, &mut lines, &mut cols); }
            ':' => { i += 1; col += 1; push(Tok::Colon, tl, tc, &mut toks, &mut lines, &mut cols); }
            '=' => { i += 1; col += 1; push(Tok::Eq, tl, tc, &mut toks, &mut lines, &mut cols); }
            c if c.is_ascii_digit() => {
                let mut s = String::new();
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    s.push(chars[i]); i += 1; col += 1;
                }
                push(Tok::Num(s.parse().unwrap()), tl, tc, &mut toks, &mut lines, &mut cols);
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let mut s = String::new();
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    s.push(chars[i]); i += 1; col += 1;
                }
                let t = match s.as_str() {
                    "fn" => Tok::Fn,
                    _ => Tok::Ident(s),
                };
                push(t, tl, tc, &mut toks, &mut lines, &mut cols);
            }
            _ => die(&format!("unexpected character: {}", c)),
        }
    }
    Lexed { toks, cols, lines }
}

#[derive(Debug)]
enum Expr {
    Num(f64),
    Arr(Vec<Expr>),
    Var(String),
    Neg(Box<Expr>),
    Bin(Op, Box<Expr>, Box<Expr>),
    Let(String, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
    Seq(Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone, Copy)]
enum Op { Add, Sub, Mul, Div }

#[derive(Debug)]
struct Decl {
    params: Vec<String>,
    body: Expr,
}

#[derive(Debug)]
struct Program {
    fns: HashMap<String, Decl>,
    main: Expr,
}

struct Parser {
    toks: Vec<Tok>,
    cols: Vec<usize>,
    lines: Vec<usize>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn peek_col(&self) -> Option<usize> {
        self.cols.get(self.pos).copied()
    }

    fn peek_line(&self) -> Option<usize> {
        self.lines.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        self.pos += 1;
        t
    }

    fn ident(&mut self, what: &str) -> String {
        match self.bump() {
            Some(Tok::Ident(s)) => s,
            t => die(&format!("expected {}, got {:?}", what, t)),
        }
    }

    fn expect(&mut self, want: Tok, what: &str) {
        let got = self.bump();
        if got.as_ref() != Some(&want) {
            die(&format!("expected {}, got {:?}", what, got));
        }
    }

    fn program(&mut self) -> Program {
        let mut fns = HashMap::new();
        let mut mains: Vec<Expr> = Vec::new();
        while self.peek().is_some() {
            if matches!(self.peek(), Some(Tok::Fn)) {
                let (name, decl) = self.decl();
                if fns.insert(name.clone(), decl).is_some() {
                    die(&format!("duplicate function: {}", name));
                }
            } else {
                let indent = self.peek_col().unwrap();
                mains.push(self.body(indent));
            }
        }
        let main = mains.into_iter()
            .reduce(|a, b| Expr::Seq(Box::new(a), Box::new(b)))
            .unwrap_or_else(|| die("program has no expressions"));
        Program { fns, main }
    }

    fn decl(&mut self) -> (String, Decl) {
        self.bump();
        let name = self.ident("function name");
        self.expect(Tok::LParen, "'(' after function name");
        let mut params = Vec::new();
        if !matches!(self.peek(), Some(Tok::RParen)) {
            params.push(self.ident("parameter"));
            while matches!(self.peek(), Some(Tok::Comma)) {
                self.bump();
                params.push(self.ident("parameter"));
            }
        }
        self.expect(Tok::RParen, "')' or ','");
        self.expect(Tok::Colon, "':' after parameters");
        if params.is_empty() {
            die(&format!("function {} has no parameters; use a binding for constants", name));
        }
        let body_indent = self.peek_col().unwrap_or(1);
        let body = self.body(body_indent);
        (name, Decl { params, body })
    }

    fn body(&mut self, indent: usize) -> Expr {
        if let Some(Tok::Ident(_)) = self.peek() {
            if matches!(self.toks.get(self.pos + 1), Some(Tok::Eq)) {
                let name = self.ident("binding name");
                self.bump();
                let value = self.add_sub();
                if self.body_continues(indent) {
                    let rest = self.body(indent);
                    return Expr::Let(name, Box::new(value), Box::new(rest));
                }
                die(&format!("binding {} has no body expression", name));
            }
        }
        let e = self.add_sub();
        if self.body_continues(indent) {
            let rest = self.body(indent);
            Expr::Seq(Box::new(e), Box::new(rest))
        } else {
            e
        }
    }

    fn body_continues(&self, indent: usize) -> bool {
        match self.peek() {
            None => false,
            Some(Tok::Fn) => false,
            Some(_) => {
                let curr_line = self.peek_line().unwrap();
                let curr_col = self.peek_col().unwrap();
                let prev_line = if self.pos > 0 { self.lines[self.pos - 1] } else { 0 };
                if curr_line == prev_line {
                    true
                } else {
                    curr_col >= indent
                }
            }
        }
    }

    fn add_sub(&mut self) -> Expr {
        let mut lhs = self.mul_div();
        loop {
            let op = match self.peek() {
                Some(Tok::Plus) => Op::Add,
                Some(Tok::Minus) => Op::Sub,
                _ => break,
            };
            self.bump();
            let rhs = self.mul_div();
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        lhs
    }

    fn mul_div(&mut self) -> Expr {
        let mut lhs = self.unary();
        loop {
            let op = match self.peek() {
                Some(Tok::Star) => Op::Mul,
                Some(Tok::Slash) => Op::Div,
                _ => break,
            };
            self.bump();
            let rhs = self.unary();
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        lhs
    }

    fn unary(&mut self) -> Expr {
        if matches!(self.peek(), Some(Tok::Minus)) {
            self.bump();
            Expr::Neg(Box::new(self.unary()))
        } else {
            self.atom()
        }
    }

    fn atom(&mut self) -> Expr {
        match self.bump() {
            Some(Tok::Num(n)) => Expr::Num(n),
            Some(Tok::Ident(s)) => {
                if matches!(self.peek(), Some(Tok::LParen)) {
                    self.bump();
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Some(Tok::RParen)) {
                        args.push(self.add_sub());
                        while matches!(self.peek(), Some(Tok::Comma)) {
                            self.bump();
                            args.push(self.add_sub());
                        }
                    }
                    self.expect(Tok::RParen, "')' or ','");
                    Expr::Call(s, args)
                } else {
                    Expr::Var(s)
                }
            }
            Some(Tok::LParen) => {
                let e = self.add_sub();
                self.expect(Tok::RParen, "')'");
                e
            }
            Some(Tok::LBracket) => {
                let mut elems = Vec::new();
                if !matches!(self.peek(), Some(Tok::RBracket)) {
                    elems.push(self.add_sub());
                    while matches!(self.peek(), Some(Tok::Comma)) {
                        self.bump();
                        elems.push(self.add_sub());
                    }
                }
                self.expect(Tok::RBracket, "']' or ','");
                Expr::Arr(elems)
            }
            t => die(&format!("unexpected token: {:?}", t)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Dtype { F32, F64, I1 }

impl Dtype {
    fn name(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F64 => "f64",
            Dtype::I1 => "i1",
        }
    }
}

fn tensor_type(shape: &[usize], dtype: Dtype) -> String {
    let dims: String = shape.iter().map(|d| format!("{}x", d)).collect();
    format!("tensor<{}{}>", dims, dtype.name())
}

fn mlir_float(n: f64) -> String {
    let s = format!("{:?}", n);
    if s.contains('e') && !s.contains('.') {
        s.replace('e', ".0e")
    } else {
        s
    }
}

fn axis_arg(e: &Expr, shape: &[usize]) -> usize {
    let n = match e {
        Expr::Num(n) => *n,
        _ => die("reduction axis must be a number literal"),
    };
    if n.fract() != 0.0 || n < 0.0 || n as usize >= shape.len() {
        die(&format!("reduction axis {} out of range for shape {:?}", n, shape));
    }
    n as usize
}

#[derive(Debug, Clone)]
struct Val {
    id: usize,
    shape: Vec<usize>,
    dtype: Dtype,
}

#[derive(Debug, Clone)]
enum OpKind {
    Constant(f64),
    Ewise(String),
    Unary(String),
    Convert,
    Broadcast(Vec<usize>),
    Reshape,
    Concat,
    Reduce(Vec<usize>),
    Dot(Vec<usize>, Vec<usize>),
    Compare(String),
    Select,
    Slice(usize, usize),
}

#[derive(Debug, Clone)]
struct Node {
    kind: OpKind,
    inputs: Vec<usize>,
    shape: Vec<usize>,
    dtype: Dtype,
}

struct Tracer {
    nodes: Vec<Node>,
    prints: Vec<Val>,
}

impl Tracer {
    fn emit(&mut self, kind: OpKind, inputs: Vec<usize>, shape: Vec<usize>, dtype: Dtype) -> Val {
        let id = self.nodes.len();
        self.nodes.push(Node { kind, inputs, shape: shape.clone(), dtype });
        Val { id, shape, dtype }
    }

    fn val(&self, id: usize) -> Val {
        let node = &self.nodes[id];
        Val { id, shape: node.shape.clone(), dtype: node.dtype }
    }

    fn constant(&mut self, n: f64, dtype: Dtype) -> Val {
        self.emit(OpKind::Constant(n), vec![], vec![], dtype)
    }

    fn convert(&mut self, v: &Val, dtype: Dtype) -> Val {
        if v.dtype == dtype {
            return v.clone();
        }
        self.emit(OpKind::Convert, vec![v.id], v.shape.clone(), dtype)
    }

    fn broadcast(&mut self, v: &Val, shape: &[usize]) -> Val {
        let offset = shape.len() - v.shape.len();
        let dims: Vec<usize> = (offset..shape.len()).collect();
        self.broadcast_along(v, shape, dims)
    }

    fn broadcast_along(&mut self, v: &Val, shape: &[usize], dims: Vec<usize>) -> Val {
        self.emit(OpKind::Broadcast(dims), vec![v.id], shape.to_vec(), v.dtype)
    }

    fn unary(&mut self, name: &str, v: &Val) -> Val {
        self.emit(OpKind::Unary(name.to_string()), vec![v.id], v.shape.clone(), v.dtype)
    }

    fn reshape(&mut self, v: &Val, shape: Vec<usize>) -> Val {
        self.emit(OpKind::Reshape, vec![v.id], shape, v.dtype)
    }

    fn compare(&mut self, dir: &str, a: &Val, b: &Val) -> Val {
        self.emit(OpKind::Compare(dir.to_string()), vec![a.id, b.id], a.shape.clone(), Dtype::I1)
    }

    fn select(&mut self, pred: &Val, on_true: &Val, on_false: &Val) -> Val {
        self.emit(OpKind::Select, vec![pred.id, on_true.id, on_false.id], on_true.shape.clone(), on_true.dtype)
    }

    fn zeros(&mut self, shape: &[usize], dtype: Dtype) -> Val {
        let zero = self.constant(0.0, dtype);
        if shape.is_empty() { zero } else { self.broadcast(&zero, shape) }
    }

    fn zeros_like(&mut self, v: &Val) -> Val {
        self.zeros(&v.shape.clone(), v.dtype)
    }

    fn backward(&mut self, y: &Val, x: &Val) -> Val {
        let mut cot: HashMap<usize, Val> = HashMap::new();
        let seed = self.constant(1.0, y.dtype);
        cot.insert(y.id, seed);
        for id in (0..=y.id).rev() {
            let Some(g) = cot.get(&id).cloned() else { continue };
            for (input_id, contribution) in self.vjp(id, &g) {
                let merged = match cot.remove(&input_id) {
                    Some(prev) => self.ewise("add", prev, contribution),
                    None => contribution,
                };
                cot.insert(input_id, merged);
            }
        }
        match cot.remove(&x.id) {
            Some(v) => v,
            None => self.zeros_like(x),
        }
    }

    fn vjp(&mut self, id: usize, g: &Val) -> Vec<(usize, Val)> {
        let node = self.nodes[id].clone();
        let out = self.val(id);
        let ins: Vec<Val> = node.inputs.iter().map(|&i| self.val(i)).collect();
        match &node.kind {
            OpKind::Constant(_) | OpKind::Compare(_) => vec![],
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
            OpKind::Concat => {
                let mut contribs = Vec::new();
                for (i, &input_id) in node.inputs.iter().enumerate() {
                    let shape = self.nodes[input_id].shape.clone();
                    let row = self.emit(OpKind::Slice(i, i + 1), vec![g.id], shape, g.dtype);
                    contribs.push((input_id, row));
                }
                contribs
            }
            OpKind::Reduce(axes) => {
                let kept: Vec<usize> = (0..ins[0].shape.len()).filter(|d| !axes.contains(d)).collect();
                let da = self.broadcast_along(g, &ins[0].shape.clone(), kept);
                vec![(ins[0].id, da)]
            }
            OpKind::Dot(_, _) => {
                let (a, b) = (ins[0].clone(), ins[1].clone());
                let (da, db) = match (a.shape.len(), b.shape.len()) {
                    (2, 2) => (self.dot(g, &b, vec![1], vec![1]), self.dot(&a, g, vec![0], vec![0])),
                    (1, 2) => (self.dot(&b, g, vec![1], vec![0]), self.dot(&a, g, vec![], vec![])),
                    (2, 1) => (self.dot(g, &b, vec![], vec![]), self.dot(&a, g, vec![0], vec![0])),
                    _ => (self.ewise("multiply", g.clone(), b.clone()), self.ewise("multiply", g.clone(), a.clone())),
                };
                vec![(a.id, da), (b.id, db)]
            }
            OpKind::Select => {
                let zero = self.zeros_like(g);
                let dt = self.select(&ins[0], g, &zero);
                let df = self.select(&ins[0], &zero, g);
                vec![(ins[1].id, dt), (ins[2].id, df)]
            }
            OpKind::Slice(start, limit) => {
                let in_shape = ins[0].shape.clone();
                let mut parts = Vec::new();
                if *start > 0 {
                    let mut shape = vec![*start];
                    shape.extend(&in_shape[1..]);
                    parts.push(self.zeros(&shape, g.dtype).id);
                }
                parts.push(g.id);
                if *limit < in_shape[0] {
                    let mut shape = vec![in_shape[0] - *limit];
                    shape.extend(&in_shape[1..]);
                    parts.push(self.zeros(&shape, g.dtype).id);
                }
                let da = if parts.len() == 1 {
                    g.clone()
                } else {
                    self.emit(OpKind::Concat, parts, in_shape, g.dtype)
                };
                vec![(ins[0].id, da)]
            }
        }
    }

    fn ewise(&mut self, name: &str, a: Val, b: Val) -> Val {
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

    fn binop(&mut self, op: Op, a: Val, b: Val) -> Val {
        let name = match op {
            Op::Add => "add",
            Op::Sub => "subtract",
            Op::Mul => "multiply",
            Op::Div => "divide",
        };
        self.ewise(name, a, b)
    }

    fn matmul(&mut self, a: Val, b: Val) -> Val {
        if a.dtype != b.dtype {
            die(&format!("matmul dtype mismatch: {} vs {}", a.dtype.name(), b.dtype.name()));
        }
        if a.shape.is_empty() || a.shape.len() > 2 || b.shape.is_empty() || b.shape.len() > 2 {
            die(&format!("matmul supports rank 1 and 2, got {:?} vs {:?}", a.shape, b.shape));
        }
        if a.shape[a.shape.len() - 1] != b.shape[0] {
            die(&format!("matmul contraction mismatch: {:?} vs {:?}", a.shape, b.shape));
        }
        self.dot(&a, &b, vec![a.shape.len() - 1], vec![0])
    }

    fn dot(&mut self, a: &Val, b: &Val, lc: Vec<usize>, rc: Vec<usize>) -> Val {
        let mut shape: Vec<usize> = a.shape.iter().enumerate()
            .filter(|(i, _)| !lc.contains(i))
            .map(|(_, &d)| d)
            .collect();
        shape.extend(b.shape.iter().enumerate()
            .filter(|(i, _)| !rc.contains(i))
            .map(|(_, &d)| d));
        self.emit(OpKind::Dot(lc, rc), vec![a.id, b.id], shape, a.dtype)
    }

    fn reduce_sum(&mut self, v: &Val, axes: &[usize]) -> Val {
        if axes.is_empty() {
            return v.clone();
        }
        let init = self.constant(0.0, v.dtype);
        let out_shape: Vec<usize> = v.shape.iter().enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, &d)| d)
            .collect();
        self.emit(OpKind::Reduce(axes.to_vec()), vec![v.id, init.id], out_shape, v.dtype)
    }

    fn stack(&mut self, vals: Vec<Val>) -> Val {
        let inner_shape = vals[0].shape.clone();
        let dtype = vals[0].dtype;
        for v in &vals[1..] {
            if v.shape != inner_shape {
                die(&format!("array literal has inconsistent shapes: {:?} vs {:?}", inner_shape, v.shape));
            }
            if v.dtype != dtype {
                die(&format!("array literal has inconsistent dtypes: {} vs {}", dtype.name(), v.dtype.name()));
            }
        }
        let mut row_shape = vec![1];
        row_shape.extend(&inner_shape);
        let rows: Vec<Val> = vals.iter().map(|v| self.reshape(v, row_shape.clone())).collect();
        if rows.len() == 1 {
            return rows.into_iter().next().unwrap();
        }
        let mut shape = vec![rows.len()];
        shape.extend(&inner_shape);
        let ids: Vec<usize> = rows.iter().map(|r| r.id).collect();
        self.emit(OpKind::Concat, ids, shape, dtype)
    }

    fn trace(&mut self, e: &Expr, env: &HashMap<String, Val>, fns: &HashMap<String, Decl>) -> Val {
        match e {
            Expr::Num(n) => self.constant(*n, Dtype::F64),
            Expr::Neg(inner) => {
                let v = self.trace(inner, env, fns);
                self.unary("negate", &v)
            }
            Expr::Arr(elems) => {
                if elems.is_empty() {
                    die("empty array literal");
                }
                let vals: Vec<Val> = elems.iter().map(|el| self.trace(el, env, fns)).collect();
                self.stack(vals)
            }
            Expr::Var(s) => {
                if let Some(v) = env.get(s) {
                    v.clone()
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
                if let Some(decl) = fns.get(name) {
                    if decl.params.len() != args.len() {
                        die(&format!("arity mismatch: {} expects {} args, got {}",
                                     name, decl.params.len(), args.len()));
                    }
                    let mut env2 = HashMap::new();
                    for (param, arg) in decl.params.iter().zip(args.iter()) {
                        env2.insert(param.clone(), self.trace(arg, env, fns));
                    }
                    self.trace(&decl.body, &env2, fns)
                } else {
                    self.builtin(name, args, env, fns)
                }
            }
        }
    }

    fn builtin(&mut self, name: &str, args: &[Expr], env: &HashMap<String, Val>, fns: &HashMap<String, Decl>) -> Val {
        match name {
            "sum" | "mean" => {
                if args.is_empty() || args.len() > 2 {
                    die(&format!("{} expects 1 or 2 args, got {}", name, args.len()));
                }
                let v = self.trace(&args[0], env, fns);
                let axes: Vec<usize> = if args.len() == 2 {
                    vec![axis_arg(&args[1], &v.shape)]
                } else {
                    (0..v.shape.len()).collect()
                };
                let total = self.reduce_sum(&v, &axes);
                if name == "sum" {
                    return total;
                }
                let count: usize = axes.iter().map(|&d| v.shape[d]).product();
                let denom = self.constant(count as f64, v.dtype);
                self.ewise("divide", total, denom)
            }
            "print" => {
                if args.len() != 1 {
                    die(&format!("print expects 1 arg, got {}", args.len()));
                }
                let v = self.trace(&args[0], env, fns);
                self.prints.push(v.clone());
                v
            }
            "f32" | "f64" => {
                if args.len() != 1 {
                    die(&format!("{} expects 1 arg, got {}", name, args.len()));
                }
                let dtype = if name == "f32" { Dtype::F32 } else { Dtype::F64 };
                let v = self.trace(&args[0], env, fns);
                self.convert(&v, dtype)
            }
            "exp" | "log" | "tanh" | "sqrt" => {
                if args.len() != 1 {
                    die(&format!("{} expects 1 arg, got {}", name, args.len()));
                }
                let v = self.trace(&args[0], env, fns);
                let op = if name == "exp" { "exponential" } else { name };
                self.unary(op, &v)
            }
            "max" | "min" => {
                if args.len() != 2 {
                    die(&format!("{} expects 2 args, got {}", name, args.len()));
                }
                let a = self.trace(&args[0], env, fns);
                let b = self.trace(&args[1], env, fns);
                let op = if name == "max" { "maximum" } else { "minimum" };
                self.ewise(op, a, b)
            }
            "matmul" => {
                if args.len() != 2 {
                    die(&format!("matmul expects 2 args, got {}", args.len()));
                }
                let a = self.trace(&args[0], env, fns);
                let b = self.trace(&args[1], env, fns);
                self.matmul(a, b)
            }
            "grad" => {
                let fname = match args.first() {
                    Some(Expr::Var(s)) => s.clone(),
                    _ => die("grad expects a function name as its first argument"),
                };
                let decl = fns.get(&fname)
                    .unwrap_or_else(|| die(&format!("undefined function: {}", fname)));
                if args.len() - 1 != decl.params.len() {
                    die(&format!("grad({}) expects {} args after the function name, got {}",
                                 fname, decl.params.len(), args.len() - 1));
                }
                let vals: Vec<Val> = args[1..].iter().map(|a| self.trace(a, env, fns)).collect();
                let mut env2 = HashMap::new();
                for (param, v) in decl.params.iter().zip(&vals) {
                    env2.insert(param.clone(), v.clone());
                }
                let y = self.trace(&decl.body, &env2, fns);
                if !y.shape.is_empty() {
                    die(&format!("grad requires a scalar-valued function; {} returned shape {:?}", fname, y.shape));
                }
                self.backward(&y, &vals[0])
            }
            _ => die(&format!("undefined function: {}", name)),
        }
    }
}

fn join(xs: &[usize]) -> String {
    xs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")
}

fn node_text(node: &Node, nodes: &[Node]) -> String {
    let t = |i: usize| tensor_type(&nodes[node.inputs[i]].shape, nodes[node.inputs[i]].dtype);
    let arg = |i: usize| format!("%{}", node.inputs[i]);
    let out = tensor_type(&node.shape, node.dtype);
    match &node.kind {
        OpKind::Constant(n) => format!("stablehlo.constant dense<{}> : {}", mlir_float(*n), out),
        OpKind::Ewise(name) => format!("stablehlo.{} {}, {} : {}", name, arg(0), arg(1), out),
        OpKind::Unary(name) => format!("stablehlo.{} {} : {}", name, arg(0), out),
        OpKind::Convert => format!("stablehlo.convert {} : ({}) -> {}", arg(0), t(0), out),
        OpKind::Broadcast(dims) => format!(
            "stablehlo.broadcast_in_dim {}, dims = [{}] : ({}) -> {}",
            arg(0), join(dims), t(0), out
        ),
        OpKind::Reshape => format!("stablehlo.reshape {} : ({}) -> {}", arg(0), t(0), out),
        OpKind::Concat => {
            let operands: Vec<String> = (0..node.inputs.len()).map(arg).collect();
            let in_types: Vec<String> = (0..node.inputs.len()).map(t).collect();
            format!(
                "stablehlo.concatenate {}, dim = 0 : ({}) -> {}",
                operands.join(", "), in_types.join(", "), out
            )
        }
        OpKind::Reduce(axes) => format!(
            "stablehlo.reduce({} init: {}) applies stablehlo.add across dimensions = [{}] : ({}, {}) -> {}",
            arg(0), arg(1), join(axes), t(0), t(1), out
        ),
        OpKind::Dot(lc, rc) => format!(
            "stablehlo.dot_general {}, {}, contracting_dims = [{}] x [{}] : ({}, {}) -> {}",
            arg(0), arg(1), join(lc), join(rc), t(0), t(1), out
        ),
        OpKind::Compare(dir) => format!(
            "stablehlo.compare {}, {}, {} : ({}, {}) -> {}",
            dir, arg(0), arg(1), t(0), t(1), out
        ),
        OpKind::Select => format!(
            "stablehlo.select {}, {}, {} : {}, {}",
            arg(0), arg(1), arg(2), t(0), out
        ),
        OpKind::Slice(start, limit) => {
            let mut ranges = vec![format!("{}:{}", start, limit)];
            ranges.extend(node.shape[1..].iter().map(|&d| format!("0:{}", d)));
            format!("stablehlo.slice {} [{}] : ({}) -> {}", arg(0), ranges.join(", "), t(0), out)
        }
    }
}

fn build_module(tracer: &Tracer, outputs: &[Val]) -> String {
    let types: Vec<String> = outputs.iter().map(|v| tensor_type(&v.shape, v.dtype)).collect();
    let names: Vec<String> = outputs.iter().map(|v| format!("%{}", v.id)).collect();
    let signature = if types.is_empty() {
        String::new()
    } else {
        format!(" -> ({})", types.join(", "))
    };
    let ret = if names.is_empty() {
        "    return\n".to_string()
    } else {
        format!("    return {} : {}\n", names.join(", "), types.join(", "))
    };
    let mut s = String::new();
    s.push_str("module {\n");
    s.push_str(&format!("  func.func @main(){} {{\n", signature));
    for (id, node) in tracer.nodes.iter().enumerate() {
        s.push_str(&format!("    %{} = {}\n", id, node_text(node, &tracer.nodes)));
    }
    s.push_str(&ret);
    s.push_str("  }\n}\n");
    s
}

#[derive(Debug, Clone)]
enum TensorData {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

#[derive(Debug, Clone)]
struct Tensor {
    data: TensorData,
    shape: Vec<usize>,
}

impl Tensor {
    fn dtype(&self) -> &'static str {
        match &self.data {
            TensorData::F32(_) => "f32",
            TensorData::F64(_) => "f64",
        }
    }
}

fn host_tensor(h: HostBuffer) -> Tensor {
    let shape: Vec<usize> = h.dims().iter().map(|&d| d as usize).collect();
    match h {
        HostBuffer::F32(b) => Tensor { data: TensorData::F32(b.data().to_vec()), shape },
        HostBuffer::F64(b) => Tensor { data: TensorData::F64(b.data().to_vec()), shape },
        _ => die("unexpected output dtype from XLA"),
    }
}

fn execute(mlir: &str) -> Vec<Tensor> {
    let plugin_path = plugin_path();
    let api = pjrt::plugin(&plugin_path)
        .load()
        .unwrap_or_else(|e| die(&format!("cannot load PJRT plugin at {}: {}", plugin_path, e)));
    let client = Client::builder(&api)
        .build()
        .unwrap_or_else(|e| die(&format!("cannot create PJRT client: {}", e)));
    let program = pjrt::Program::new(MLIR, mlir.as_bytes());
    let executable = LoadedExecutable::builder(&client, &program)
        .build()
        .unwrap_or_else(|e| die(&format!("XLA compilation failed: {}", e)));
    let results = executable
        .execution(())
        .run_sync()
        .unwrap_or_else(|e| die(&format!("execution failed: {}", e)));
    results[0]
        .iter()
        .map(|b| {
            let h = b.to_host_sync(None)
                .unwrap_or_else(|e| die(&format!("device-to-host transfer failed: {}", e)));
            host_tensor(h)
        })
        .collect()
}

fn format_typed<T: std::fmt::Display>(data: &[T], shape: &[usize]) -> String {
    fn rec<T: std::fmt::Display>(data: &[T], shape: &[usize]) -> String {
        if shape.is_empty() {
            return format!("{}", &data[0]);
        }
        let inner: usize = shape[1..].iter().product::<usize>().max(1);
        let parts: Vec<String> = (0..shape[0])
            .map(|i| rec(&data[i * inner..(i + 1) * inner], &shape[1..]))
            .collect();
        format!("[{}]", parts.join(", "))
    }
    rec(data, shape)
}

fn format_tensor(t: &Tensor) -> String {
    let values = match &t.data {
        TensorData::F32(v) => format_typed(v, &t.shape),
        TensorData::F64(v) => format_typed(v, &t.shape),
    };
    format!("{} : {}", values, t.dtype())
}

fn die(msg: &str) -> ! {
    eprintln!("{}", msg);
    exit(1);
}

const USAGE: &str = "usage: vector <command>

  run <file.vec>      compile and execute
  build <file.vec>    print StableHLO to stdout
  setup               download the PJRT CPU plugin to ~/.vector
  version             print version";

fn home() -> String {
    env::var("HOME").unwrap_or_else(|_| die("HOME is not set"))
}

fn plugin_file() -> &'static str {
    if cfg!(target_os = "macos") { "libpjrt_cpu.dylib" } else { "libpjrt_cpu.so" }
}

fn plugin_path() -> String {
    if let Ok(p) = env::var("PJRT_PLUGIN_PATH") {
        return p;
    }
    let path = format!("{}/.vector/{}", home(), plugin_file());
    if fs::metadata(&path).is_err() {
        die(&format!("PJRT plugin not found at {}; run `vector setup` or set PJRT_PLUGIN_PATH", path));
    }
    path
}

fn compile(path: &str) -> String {
    let src = fs::read_to_string(path)
        .unwrap_or_else(|e| die(&format!("cannot read file: {}", e)));
    let lexed = lex(&src);
    let mut p = Parser { toks: lexed.toks, cols: lexed.cols, lines: lexed.lines, pos: 0 };
    let prog = p.program();
    let mut tracer = Tracer { nodes: Vec::new(), prints: Vec::new() };
    tracer.trace(&prog.main, &HashMap::new(), &prog.fns);
    let outputs = tracer.prints.clone();
    build_module(&tracer, &outputs)
}

fn run(path: &str) {
    for tensor in execute(&compile(path)) {
        println!("{}", format_tensor(&tensor));
    }
}

fn setup() {
    let platform = match (env::consts::OS, env::consts::ARCH) {
        ("macos", "aarch64") => "darwin-arm64",
        ("macos", "x86_64") => "darwin-amd64",
        ("linux", "aarch64") => "linux-arm64",
        ("linux", "x86_64") => "linux-amd64",
        (os, arch) => die(&format!("no prebuilt PJRT CPU plugin for {}-{}", os, arch)),
    };
    let dir = format!("{}/.vector", home());
    fs::create_dir_all(&dir).unwrap_or_else(|e| die(&format!("cannot create {}: {}", dir, e)));
    let url = format!(
        "https://github.com/zml/pjrt-artifacts/releases/latest/download/pjrt-cpu_{}.tar.gz",
        platform
    );
    println!("downloading {}", url);
    let status = Command::new("sh")
        .arg("-c")
        .arg(format!("curl -fL --progress-bar {} | tar xz -C {}", url, dir))
        .status()
        .unwrap_or_else(|e| die(&format!("cannot run curl: {}", e)));
    if !status.success() {
        die("plugin download failed");
    }
    let path = format!("{}/{}", dir, plugin_file());
    if fs::metadata(&path).is_err() {
        die(&format!("download completed but {} is missing", path));
    }
    println!("installed {}", path);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("run") if args.len() == 3 => run(&args[2]),
        Some("build") if args.len() == 3 => print!("{}", compile(&args[2])),
        Some("setup") if args.len() == 2 => setup(),
        Some("version") if args.len() == 2 => println!("vector {}", env!("CARGO_PKG_VERSION")),
        Some("help") => println!("{}", USAGE),
        _ => die(USAGE),
    }
}
