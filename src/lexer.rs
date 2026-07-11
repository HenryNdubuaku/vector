use crate::die;

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Num(f64),
    Str(String),
    Ident(String),
    Fn,
    For,
    In,
    DotDot,
    Eq,
    Lt,
    Gt,
    Le,
    Ge,
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Dot,
}

pub struct Lexed {
    pub toks: Vec<Tok>,
    pub cols: Vec<usize>,
    pub lines: Vec<usize>,
}

pub fn lex(src: &str) -> Lexed {
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
            '-' => { i += 1; col += 1; push(Tok::Minus, tl, tc, &mut toks, &mut lines, &mut cols); }
            '#' => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1; col += 1;
                }
            }
            '+' => { i += 1; col += 1; push(Tok::Plus, tl, tc, &mut toks, &mut lines, &mut cols); }
            '*' => { i += 1; col += 1; push(Tok::Star, tl, tc, &mut toks, &mut lines, &mut cols); }
            '/' => { i += 1; col += 1; push(Tok::Slash, tl, tc, &mut toks, &mut lines, &mut cols); }
            '(' => { i += 1; col += 1; push(Tok::LParen, tl, tc, &mut toks, &mut lines, &mut cols); }
            ')' => { i += 1; col += 1; push(Tok::RParen, tl, tc, &mut toks, &mut lines, &mut cols); }
            '[' => { i += 1; col += 1; push(Tok::LBracket, tl, tc, &mut toks, &mut lines, &mut cols); }
            ']' => { i += 1; col += 1; push(Tok::RBracket, tl, tc, &mut toks, &mut lines, &mut cols); }
            '{' => { i += 1; col += 1; push(Tok::LBrace, tl, tc, &mut toks, &mut lines, &mut cols); }
            '}' => { i += 1; col += 1; push(Tok::RBrace, tl, tc, &mut toks, &mut lines, &mut cols); }
            ',' => { i += 1; col += 1; push(Tok::Comma, tl, tc, &mut toks, &mut lines, &mut cols); }
            ':' => { i += 1; col += 1; push(Tok::Colon, tl, tc, &mut toks, &mut lines, &mut cols); }
            '=' => { i += 1; col += 1; push(Tok::Eq, tl, tc, &mut toks, &mut lines, &mut cols); }
            '<' => {
                i += 1; col += 1;
                if chars.get(i) == Some(&'=') {
                    i += 1; col += 1;
                    push(Tok::Le, tl, tc, &mut toks, &mut lines, &mut cols);
                } else {
                    push(Tok::Lt, tl, tc, &mut toks, &mut lines, &mut cols);
                }
            }
            '>' => {
                i += 1; col += 1;
                if chars.get(i) == Some(&'=') {
                    i += 1; col += 1;
                    push(Tok::Ge, tl, tc, &mut toks, &mut lines, &mut cols);
                } else {
                    push(Tok::Gt, tl, tc, &mut toks, &mut lines, &mut cols);
                }
            }
            '.' => {
                if chars.get(i + 1) == Some(&'.') {
                    i += 2; col += 2;
                    push(Tok::DotDot, tl, tc, &mut toks, &mut lines, &mut cols);
                } else {
                    i += 1; col += 1;
                    push(Tok::Dot, tl, tc, &mut toks, &mut lines, &mut cols);
                }
            }
            '"' => {
                i += 1; col += 1;
                let mut s = String::new();
                while i < chars.len() && chars[i] != '"' && chars[i] != '\n' {
                    s.push(chars[i]); i += 1; col += 1;
                }
                if i >= chars.len() || chars[i] != '"' {
                    die("unterminated string literal");
                }
                i += 1; col += 1;
                push(Tok::Str(s), tl, tc, &mut toks, &mut lines, &mut cols);
            }
            c if c.is_ascii_digit() => {
                let mut s = String::new();
                while i < chars.len()
                    && (chars[i].is_ascii_digit()
                        || (chars[i] == '.' && chars.get(i + 1) != Some(&'.'))) {
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
                    "for" => Tok::For,
                    "in" => Tok::In,
                    _ => Tok::Ident(s),
                };
                push(t, tl, tc, &mut toks, &mut lines, &mut cols);
            }
            _ => die(&format!("unexpected character: {}", c)),
        }
    }
    Lexed { toks, cols, lines }
}
