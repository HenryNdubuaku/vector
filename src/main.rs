use std::collections::HashMap;
use std::env;
use std::fs;
use std::process::exit;

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
        while matches!(self.peek(), Some(Tok::Fn)) {
            let (name, decl) = self.decl();
            if fns.insert(name.clone(), decl).is_some() {
                die(&format!("duplicate function: {}", name));
            }
        }
        let main_indent = self.peek_col().unwrap_or(1);
        let main = self.body(main_indent);
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
        let mut lhs = self.atom();
        loop {
            let op = match self.peek() {
                Some(Tok::Star) => Op::Mul,
                Some(Tok::Slash) => Op::Div,
                _ => break,
            };
            self.bump();
            let rhs = self.atom();
            lhs = Expr::Bin(op, Box::new(lhs), Box::new(rhs));
        }
        lhs
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

trait Numeric:
    Copy
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::iter::Sum
    + std::fmt::Display {}
impl Numeric for f32 {}
impl Numeric for f64 {}

fn apply<T: Numeric>(x: T, y: T, op: Op) -> T {
    match op {
        Op::Add => x + y,
        Op::Sub => x - y,
        Op::Mul => x * y,
        Op::Div => x / y,
    }
}

fn ewise_typed<T: Numeric>(av: &[T], bv: &[T], a_shape: &[usize], b_shape: &[usize], op: Op) -> (Vec<T>, Vec<usize>) {
    if a_shape == b_shape {
        (av.iter().zip(bv.iter()).map(|(x, y)| apply(*x, *y, op)).collect(), a_shape.to_vec())
    } else if a_shape.is_empty() {
        let s = av[0];
        (bv.iter().map(|y| apply(s, *y, op)).collect(), b_shape.to_vec())
    } else if b_shape.is_empty() {
        let s = bv[0];
        (av.iter().map(|x| apply(*x, s, op)).collect(), a_shape.to_vec())
    } else {
        die(&format!("shape mismatch: {:?} vs {:?}", a_shape, b_shape));
    }
}

fn cast(t: &Tensor, dtype: &str) -> Tensor {
    let data = match (dtype, &t.data) {
        ("f32", TensorData::F32(_)) => t.data.clone(),
        ("f32", TensorData::F64(v)) => TensorData::F32(v.iter().map(|&x| x as f32).collect()),
        ("f64", TensorData::F32(v)) => TensorData::F64(v.iter().map(|&x| x as f64).collect()),
        ("f64", TensorData::F64(_)) => t.data.clone(),
        _ => die(&format!("unknown dtype: {}", dtype)),
    };
    Tensor { data, shape: t.shape.clone() }
}

fn ewise(a: &Tensor, b: &Tensor, op: Op) -> Tensor {
    let (a, b) = match (&a.data, &b.data) {
        (TensorData::F32(_), TensorData::F32(_)) => (a.clone(), b.clone()),
        (TensorData::F64(_), TensorData::F64(_)) => (a.clone(), b.clone()),
        _ => {
            if a.shape.is_empty() {
                (cast(a, b.dtype()), b.clone())
            } else if b.shape.is_empty() {
                (a.clone(), cast(b, a.dtype()))
            } else {
                die(&format!("dtype mismatch: {} vs {}", a.dtype(), b.dtype()));
            }
        }
    };
    match (&a.data, &b.data) {
        (TensorData::F32(av), TensorData::F32(bv)) => {
            let (data, shape) = ewise_typed(av, bv, &a.shape, &b.shape, op);
            Tensor { data: TensorData::F32(data), shape }
        }
        (TensorData::F64(av), TensorData::F64(bv)) => {
            let (data, shape) = ewise_typed(av, bv, &a.shape, &b.shape, op);
            Tensor { data: TensorData::F64(data), shape }
        }
        _ => unreachable!(),
    }
}

fn sum_tensor(t: &Tensor) -> Tensor {
    match &t.data {
        TensorData::F32(v) => {
            let s: f32 = v.iter().copied().sum();
            Tensor { data: TensorData::F32(vec![s]), shape: vec![] }
        }
        TensorData::F64(v) => {
            let s: f64 = v.iter().copied().sum();
            Tensor { data: TensorData::F64(vec![s]), shape: vec![] }
        }
    }
}

fn eval_arr(elems: &[Expr], env: &HashMap<String, Tensor>, fns: &HashMap<String, Decl>) -> Tensor {
    if elems.is_empty() {
        die("empty array literal");
    }
    let vals: Vec<Tensor> = elems.iter().map(|e| eval(e, env, fns)).collect();
    let inner_shape = vals[0].shape.clone();
    let inner_dtype = vals[0].dtype();
    for v in &vals[1..] {
        if v.shape != inner_shape {
            die(&format!("array literal has inconsistent shapes: {:?} vs {:?}", inner_shape, v.shape));
        }
        if v.dtype() != inner_dtype {
            die(&format!("array literal has inconsistent dtypes: {} vs {}", inner_dtype, v.dtype()));
        }
    }
    let mut shape = vec![vals.len()];
    shape.extend(&inner_shape);
    match inner_dtype {
        "f32" => {
            let mut data = Vec::new();
            for v in &vals {
                if let TensorData::F32(d) = &v.data { data.extend(d); }
            }
            Tensor { data: TensorData::F32(data), shape }
        }
        "f64" => {
            let mut data = Vec::new();
            for v in &vals {
                if let TensorData::F64(d) = &v.data { data.extend(d); }
            }
            Tensor { data: TensorData::F64(data), shape }
        }
        _ => unreachable!(),
    }
}

fn eval(e: &Expr, env: &HashMap<String, Tensor>, fns: &HashMap<String, Decl>) -> Tensor {
    match e {
        Expr::Num(n) => Tensor { data: TensorData::F64(vec![*n]), shape: vec![] },
        Expr::Arr(elems) => eval_arr(elems, env, fns),
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
            let l = eval(l, env, fns);
            let r = eval(r, env, fns);
            ewise(&l, &r, *op)
        }
        Expr::Let(name, value, body) => {
            let v = eval(value, env, fns);
            let mut env2 = env.clone();
            env2.insert(name.clone(), v);
            eval(body, &env2, fns)
        }
        Expr::Seq(first, rest) => {
            eval(first, env, fns);
            eval(rest, env, fns)
        }
        Expr::Call(name, args) => {
            if let Some(decl) = fns.get(name) {
                if decl.params.len() != args.len() {
                    die(&format!("arity mismatch: {} expects {} args, got {}",
                                 name, decl.params.len(), args.len()));
                }
                let mut env2 = HashMap::new();
                for (param, arg) in decl.params.iter().zip(args.iter()) {
                    env2.insert(param.clone(), eval(arg, env, fns));
                }
                eval(&decl.body, &env2, fns)
            } else {
                call_builtin(name, args, env, fns)
            }
        }
    }
}

fn call_builtin(name: &str, args: &[Expr], env: &HashMap<String, Tensor>, fns: &HashMap<String, Decl>) -> Tensor {
    match name {
        "sum" => {
            if args.len() != 1 {
                die(&format!("sum expects 1 arg, got {}", args.len()));
            }
            let t = eval(&args[0], env, fns);
            sum_tensor(&t)
        }
        "print" => {
            if args.len() != 1 {
                die(&format!("print expects 1 arg, got {}", args.len()));
            }
            let t = eval(&args[0], env, fns);
            println!("{}", format_tensor(&t));
            t
        }
        "f32" | "f64" => {
            if args.len() != 1 {
                die(&format!("{} expects 1 arg, got {}", name, args.len()));
            }
            let t = eval(&args[0], env, fns);
            cast(&t, name)
        }
        _ => die(&format!("undefined function: {}", name)),
    }
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

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        die("usage: vector <file.vec>");
    }
    let src = fs::read_to_string(&args[1])
        .unwrap_or_else(|e| die(&format!("cannot read file: {}", e)));
    let lexed = lex(&src);
    let mut p = Parser { toks: lexed.toks, cols: lexed.cols, lines: lexed.lines, pos: 0 };
    let prog = p.program();
    let result = eval(&prog.main, &HashMap::new(), &prog.fns);
    println!("{}", format_tensor(&result));
}
