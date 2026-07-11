use std::collections::HashMap;

use crate::die;
use crate::lexer::Tok;

#[derive(Debug, Clone)]
pub enum Expr {
    Unit,
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
    For(String, Box<Expr>, Box<Expr>, Option<Box<Expr>>, Vec<(Option<String>, Expr)>, Box<Expr>),
    Call(String, Vec<Expr>),
    Apply(Box<Expr>, Vec<Expr>),
    Seq(Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone, Copy)]
pub enum Op { Add, Sub, Mul, Div }

#[derive(Debug, Clone)]
pub struct Decl {
    pub params: Vec<String>,
    pub body: Expr,
}

#[derive(Debug, Clone)]
pub struct ModuleDecl {
    pub params: Vec<String>,
    pub init: Vec<(String, Expr)>,
    pub methods: Vec<(String, Decl)>,
}

impl ModuleDecl {
    pub fn method(&self, name: &str) -> Option<&Decl> {
        self.methods.iter().find(|(n, _)| n == name).map(|(_, d)| d)
    }
}

#[derive(Debug)]
pub struct Program {
    pub fns: HashMap<String, Decl>,
    pub modules: HashMap<String, ModuleDecl>,
    pub main: Expr,
}

pub struct Parser {
    pub repl: bool,
    pub toks: Vec<Tok>,
    pub cols: Vec<usize>,
    pub lines: Vec<usize>,
    pub pos: usize,
    pub fns: HashMap<String, Decl>,
    pub modules: HashMap<String, ModuleDecl>,
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
        if self.peek().is_none() {
            die("program has no expressions");
        }
        let main = self.body(1);
        Program {
            fns: std::mem::take(&mut self.fns),
            modules: std::mem::take(&mut self.modules),
            main,
        }
    }

    fn param_list(&mut self, allow_empty: bool, what: &str) -> Vec<String> {
        self.expect(Tok::LParen, "'('");
        let mut params = Vec::new();
        if !matches!(self.peek(), Some(Tok::RParen)) {
            params.push(self.ident("parameter"));
            while matches!(self.peek(), Some(Tok::Comma)) {
                self.bump();
                params.push(self.ident("parameter"));
            }
        }
        self.expect(Tok::RParen, "')' or ','");
        if params.is_empty() && !allow_empty {
            die(&format!("{} has no parameters; use a binding for constants", what));
        }
        params
    }

    fn module_decl(&mut self) -> (String, ModuleDecl) {
        let module_col = self.peek_col().unwrap();
        self.bump();
        let name = self.ident("module name");
        let params = self.param_list(true, &name);
        self.expect(Tok::Colon, "':' after module parameters");
        let member_col = self.peek_col().unwrap_or(0);
        if member_col <= module_col {
            die(&format!("module {} body must be indented past 'module'", name));
        }
        let mut init: Vec<(String, Expr)> = Vec::new();
        let mut methods: Vec<(String, Decl)> = Vec::new();
        while matches!(self.peek(), Some(Tok::Ident(_))) && self.peek_col() == Some(member_col) {
            let mname = self.ident("module member");
            if matches!(self.peek(), Some(Tok::Eq)) {
                self.bump();
                if methods.iter().any(|(n, _)| *n == mname) {
                    die(&format!("module {} member {} is both a method and a field", name, mname));
                }
                init.push((mname, self.expr()));
            } else if matches!(self.peek(), Some(Tok::LParen)) {
                if methods.iter().any(|(n, _)| *n == mname) || init.iter().any(|(n, _)| *n == mname) {
                    die(&format!("duplicate module member {} in {}", mname, name));
                }
                let mparams = self.param_list(false, &format!("method {}", mname));
                self.expect(Tok::Colon, "':' after method parameters");
                let body_indent = self.peek_col().unwrap_or(1);
                let body = self.body(body_indent);
                methods.push((mname, Decl { params: mparams, body }));
            } else {
                die(&format!("expected '=' or '(' after {} in module {}", mname, name));
            }
        }
        if !methods.iter().any(|(n, _)| n == "forward") {
            die(&format!("module {} must define forward", name));
        }
        (name, ModuleDecl { params, init, methods })
    }

    fn decl(&mut self) -> (String, Decl) {
        self.bump();
        let name = self.ident("function name");
        let params = self.param_list(false, &format!("function {}", name));
        self.expect(Tok::Colon, "':' after parameters");
        let body_indent = self.peek_col().unwrap_or(1);
        let body = self.body(body_indent);
        (name, Decl { params, body })
    }

    fn body(&mut self, indent: usize) -> Expr {
        loop {
            match self.peek() {
                Some(Tok::Fn) => {
                    let (name, decl) = self.decl();
                    if !self.repl && (self.modules.contains_key(&name) || self.fns.contains_key(&name)) {
                        die(&format!("duplicate function: {}", name));
                    }
                    self.fns.insert(name, decl);
                }
                Some(Tok::Module) => {
                    let (name, decl) = self.module_decl();
                    if !self.repl && (self.fns.contains_key(&name) || self.modules.contains_key(&name)) {
                        die(&format!("duplicate module: {}", name));
                    }
                    self.modules.insert(name, decl);
                }
                None => {
                    if self.repl {
                        return Expr::Unit;
                    }
                    die("expected an expression after declarations");
                }
                _ => break,
            }
        }
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
                if self.repl {
                    return Expr::Let(name, Box::new(value), Box::new(Expr::Unit));
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

    fn for_loop(&mut self, indent: usize) -> Expr {
        let for_col = self.peek_col().unwrap();
        self.bump();
        let var = self.ident("loop variable");
        self.expect(Tok::In, "'in' after loop variable");
        let start = Box::new(self.unary());
        self.expect(Tok::DotDot, "'..' in range");
        let end = Box::new(self.unary());
        let step = match self.peek() {
            Some(Tok::Ident(s)) if s == "by" && self.same_line() => {
                self.bump();
                Some(Box::new(self.unary()))
            }
            _ => None,
        };
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
            if self.repl {
                return Expr::For(var, start, end, step, stmts, Box::new(Expr::Unit));
            }
            die("for loop must be followed by an expression");
        }
        let rest = self.body(indent);
        Expr::For(var, start, end, step, stmts, Box::new(rest))
    }

    fn same_line(&self) -> bool {
        self.pos > 0 && self.peek_line() == Some(self.lines[self.pos - 1])
    }

    fn expr(&mut self) -> Expr {
        let lhs = self.add_sub();
        if !self.same_line() {
            return lhs;
        }
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
            if !self.same_line() {
                break;
            }
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
            if !self.same_line() {
                break;
            }
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
        loop {
            if matches!(self.peek(), Some(Tok::Dot)) {
                self.bump();
                let name = self.ident("field name after '.'");
                e = Expr::Field(Box::new(e), name);
            } else if matches!(self.peek(), Some(Tok::LParen))
                && self.pos > 0
                && self.peek_line() == Some(self.lines[self.pos - 1]) {
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
                e = Expr::Apply(Box::new(e), args);
            } else {
                break;
            }
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

