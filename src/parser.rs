use std::collections::HashMap;

use crate::die;
use crate::lexer::Tok;

#[derive(Debug)]
pub enum Expr {
    Num(f64),
    Str(String),
    Arr(Vec<Expr>),
    RecordLit(Vec<(String, Expr)>),
    Field(Box<Expr>, String),
    Var(String),
    Neg(Box<Expr>),
    Bin(Op, Box<Expr>, Box<Expr>),
    Cmp(String, Box<Expr>, Box<Expr>),
    Let(String, Box<Expr>, Box<Expr>),
    For(String, usize, usize, Vec<(Option<String>, Expr)>, Box<Expr>),
    Call(String, Vec<Expr>),
    Seq(Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone, Copy)]
pub enum Op { Add, Sub, Mul, Div }

#[derive(Debug)]
pub struct Decl {
    pub params: Vec<String>,
    pub body: Expr,
}

#[derive(Debug)]
pub struct Program {
    pub fns: HashMap<String, Decl>,
    pub main: Expr,
}

pub struct Parser {
    pub toks: Vec<Tok>,
    pub cols: Vec<usize>,
    pub lines: Vec<usize>,
    pub pos: usize,
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

    pub fn program(&mut self) -> Program {
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
            .reduce(join_main)
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
        if matches!(self.peek(), Some(Tok::For)) {
            return self.for_loop(indent);
        }
        if let Some(Tok::Ident(_)) = self.peek() {
            if matches!(self.toks.get(self.pos + 1), Some(Tok::Eq)) {
                let name = self.ident("binding name");
                self.bump();
                let value = self.expr();
                if self.body_continues(indent) {
                    let rest = self.body(indent);
                    return Expr::Let(name, Box::new(value), Box::new(rest));
                }
                die(&format!("binding {} has no body expression", name));
            }
        }
        let e = self.expr();
        if self.body_continues(indent) {
            let rest = self.body(indent);
            Expr::Seq(Box::new(e), Box::new(rest))
        } else {
            e
        }
    }

    fn int(&mut self, what: &str) -> usize {
        match self.bump() {
            Some(Tok::Num(n)) if n.fract() == 0.0 && n >= 0.0 => n as usize,
            t => die(&format!("expected {} (integer literal), got {:?}", what, t)),
        }
    }

    fn for_loop(&mut self, indent: usize) -> Expr {
        let for_col = self.peek_col().unwrap();
        self.bump();
        let var = self.ident("loop variable");
        self.expect(Tok::In, "'in' after loop variable");
        let start = self.int("range start");
        self.expect(Tok::DotDot, "'..' in range");
        let end = self.int("range end");
        self.expect(Tok::Colon, "':' after range");
        let body_col = self.peek_col().unwrap_or(0);
        if body_col <= for_col {
            die("for body must be indented past 'for'");
        }
        let mut stmts = Vec::new();
        loop {
            if let Some(Tok::Ident(_)) = self.peek() {
                if matches!(self.toks.get(self.pos + 1), Some(Tok::Eq)) {
                    let name = self.ident("binding name");
                    self.bump();
                    stmts.push((Some(name), self.expr()));
                    if self.body_continues(body_col) { continue; }
                    break;
                }
            }
            stmts.push((None, self.expr()));
            if self.body_continues(body_col) { continue; }
            break;
        }
        if !self.body_continues(indent) {
            die("for loop must be followed by an expression");
        }
        let rest = self.body(indent);
        Expr::For(var, start, end, stmts, Box::new(rest))
    }

    fn expr(&mut self) -> Expr {
        let lhs = self.add_sub();
        let dir = match self.peek() {
            Some(Tok::Lt) => "LT",
            Some(Tok::Gt) => "GT",
            Some(Tok::Le) => "LE",
            Some(Tok::Ge) => "GE",
            _ => return lhs,
        };
        self.bump();
        let rhs = self.add_sub();
        Expr::Cmp(dir.to_string(), Box::new(lhs), Box::new(rhs))
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
            self.postfix()
        }
    }

    fn postfix(&mut self) -> Expr {
        let mut e = self.atom();
        while matches!(self.peek(), Some(Tok::Dot)) {
            self.bump();
            let name = self.ident("field name after '.'");
            e = Expr::Field(Box::new(e), name);
        }
        e
    }

    fn atom(&mut self) -> Expr {
        match self.bump() {
            Some(Tok::Num(n)) => Expr::Num(n),
            Some(Tok::Str(s)) => Expr::Str(s),
            Some(Tok::Ident(s)) => {
                if matches!(self.peek(), Some(Tok::LParen)) {
                    self.bump();
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Some(Tok::RParen)) {
                        args.push(self.expr());
                        while matches!(self.peek(), Some(Tok::Comma)) {
                            self.bump();
                            args.push(self.expr());
                        }
                    }
                    self.expect(Tok::RParen, "')' or ','");
                    Expr::Call(s, args)
                } else {
                    Expr::Var(s)
                }
            }
            Some(Tok::LParen) => {
                let e = self.expr();
                self.expect(Tok::RParen, "')'");
                e
            }
            Some(Tok::LBracket) => {
                let mut elems = Vec::new();
                if !matches!(self.peek(), Some(Tok::RBracket)) {
                    elems.push(self.expr());
                    while matches!(self.peek(), Some(Tok::Comma)) {
                        self.bump();
                        elems.push(self.expr());
                    }
                }
                self.expect(Tok::RBracket, "']' or ','");
                Expr::Arr(elems)
            }
            Some(Tok::LBrace) => {
                let mut fields: Vec<(String, Expr)> = Vec::new();
                if !matches!(self.peek(), Some(Tok::RBrace)) {
                    loop {
                        let name = self.ident("record field name");
                        if fields.iter().any(|(k, _)| *k == name) {
                            die(&format!("duplicate record field: {}", name));
                        }
                        self.expect(Tok::Colon, "':' after field name");
                        fields.push((name, self.expr()));
                        if matches!(self.peek(), Some(Tok::Comma)) {
                            self.bump();
                            continue;
                        }
                        break;
                    }
                }
                self.expect(Tok::RBrace, "'}' or ','");
                if fields.is_empty() {
                    die("empty record literal");
                }
                Expr::RecordLit(fields)
            }
            t => die(&format!("unexpected token: {:?}", t)),
        }
    }
}

fn join_main(a: Expr, b: Expr) -> Expr {
    match a {
        Expr::Let(name, value, body) => Expr::Let(name, value, Box::new(join_main(*body, b))),
        Expr::Seq(first, rest) => Expr::Seq(first, Box::new(join_main(*rest, b))),
        Expr::For(var, start, end, stmts, rest) => Expr::For(var, start, end, stmts, Box::new(join_main(*rest, b))),
        other => Expr::Seq(Box::new(other), Box::new(b)),
    }
}
