use std::fs;

use pjrt::HostBuffer;

use crate::die;
use crate::graph::{BVal, Dtype, InputSource, OpKind, TVal};
use crate::runtime::Tensor;
use crate::safetensors::SaveSpec;
use crate::trace::Tracer;

pub struct Table {
    pub names: Vec<String>,
    pub columns: Vec<Vec<f64>>,
}

fn parse_rows(text: &str, path: &str) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quotes {
            match c {
                '"' if chars.peek() == Some(&'"') => {
                    chars.next();
                    field.push('"');
                }
                '"' => in_quotes = false,
                c => field.push(c),
            }
            continue;
        }
        match c {
            '"' if field.is_empty() => in_quotes = true,
            ',' => row.push(std::mem::take(&mut field)),
            '\r' => {}
            '\n' => {
                row.push(std::mem::take(&mut field));
                if row.len() > 1 || !row[0].is_empty() {
                    rows.push(std::mem::take(&mut row));
                } else {
                    row.clear();
                }
            }
            c => field.push(c),
        }
    }
    if in_quotes {
        die(&format!("{} has an unterminated quote", path));
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    rows
}

fn column_name(header: &str, path: &str) -> String {
    let mut s: String = header.trim().chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if s.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        s.insert(0, '_');
    }
    if s.is_empty() || s.chars().all(|c| c == '_') {
        die(&format!("{}: column header '{}' is not usable as a field name", path, header));
    }
    s
}

pub fn read_table(path: &str) -> Table {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|e| die(&format!("cannot read {}: {}", path, e)));
    let mut rows = parse_rows(&text, path);
    if rows.is_empty() {
        die(&format!("{} is empty", path));
    }
    let header = rows.remove(0);
    if rows.is_empty() {
        die(&format!("{} has no data rows", path));
    }
    let names: Vec<String> = header.iter().map(|h| column_name(h, path)).collect();
    for (i, name) in names.iter().enumerate() {
        if names[..i].contains(name) {
            die(&format!("{}: duplicate column {}", path, name));
        }
    }
    for (i, row) in rows.iter().enumerate() {
        if row.len() != names.len() {
            die(&format!("{}: row {} has {} fields, expected {}", path, i + 2, row.len(), names.len()));
        }
    }
    let mut columns = Vec::new();
    for (j, name) in names.iter().enumerate() {
        for (i, row) in rows.iter().enumerate() {
            if row[j].trim().is_empty() {
                die(&format!("{}: row {}, column {} is empty", path, i + 2, name));
            }
        }
        let parsed: Vec<Option<f64>> = rows.iter().map(|row| row[j].trim().parse().ok()).collect();
        if parsed.iter().all(|v| v.is_some()) {
            columns.push(parsed.into_iter().map(|v| v.unwrap()).collect());
        } else {
            let mut seen: Vec<&str> = Vec::new();
            let mut codes = Vec::new();
            for row in &rows {
                let cell = row[j].as_str();
                let code = seen.iter().position(|&s| s == cell).unwrap_or_else(|| {
                    seen.push(cell);
                    seen.len() - 1
                });
                codes.push(code as f64);
            }
            columns.push(codes);
        }
    }
    Table { names, columns }
}

pub fn csv_host_buffer(path: &str, name: &str, shape: &[usize]) -> HostBuffer {
    let table = read_table(path);
    let j = table.names.iter().position(|n| n == name)
        .unwrap_or_else(|| die(&format!("{} changed since compilation: column {} is missing", path, name)));
    let col = &table.columns[j];
    if shape != [col.len()] {
        die(&format!("{} changed since compilation: column {} has {} rows, expected {}",
                     path, name, col.len(), shape[0]));
    }
    let vals: Vec<f32> = col.iter().map(|&v| v as f32).collect();
    HostBuffer::from_data(vals, Some(vec![col.len() as i64]), None)
}

fn column_val(v: &TVal, what: &str) -> (BVal, usize) {
    let b = match v {
        TVal::Tensor(b) => b.clone(),
        TVal::Record(..) => die(&format!("{} expects a flat record of columns", what)),
    };
    if b.bdims != 0 {
        die(&format!("{} inside vmap isn't supported", what));
    }
    if b.val.dtype == Dtype::I1 {
        die("cannot save booleans; use where to select values");
    }
    let s = &b.val.shape;
    let vector_like = s.len() == 1 || (s.len() == 2 && (s[0] == 1 || s[1] == 1));
    if !vector_like {
        die(&format!("{} expects column vectors, got shape {:?}", what, s));
    }
    let count = s.iter().product();
    (b, count)
}

pub fn csv_save_spec(tracer: &mut Tracer, v: &TVal, path: &str) -> SaveSpec {
    let fields = match v {
        TVal::Record(_, fields) => fields.clone(),
        TVal::Tensor(_) => die("save to .csv expects a record of columns"),
    };
    let mut spec = SaveSpec {
        path: path.to_string(),
        names: Vec::new(),
        vals: Vec::new(),
        metadata: Vec::new(),
        value: TVal::Record(None, Vec::new()),
    };
    let mut rows = None;
    let mut flat = Vec::new();
    for (name, f) in &fields {
        let (b, count) = column_val(f, "save to .csv");
        if *rows.get_or_insert(count) != count {
            die(&format!("save to .csv columns differ in length: {} has {}, expected {}",
                         name, count, rows.unwrap()));
        }
        let column = tracer.reshape(&b.val, vec![count]);
        let column = tracer.convert(&column, Dtype::F32);
        spec.names.push(name.clone());
        spec.vals.push(column.clone());
        flat.push((name.clone(), TVal::Tensor(BVal { val: column, bdims: 0 })));
    }
    if spec.names.is_empty() {
        die("save to .csv expects at least one column");
    }
    spec.value = TVal::Record(None, flat);
    spec
}

pub fn write_csv(spec: &SaveSpec, tensors: &[Tensor]) {
    let columns: Vec<Vec<f64>> = tensors.iter().map(|t| t.f64_vec()).collect();
    let mut s = spec.names.join(",");
    s.push('\n');
    for i in 0..columns[0].len() {
        let row: Vec<String> = columns.iter().map(|c| format!("{}", c[i] as f32)).collect();
        s.push_str(&row.join(","));
        s.push('\n');
    }
    fs::write(&spec.path, s)
        .unwrap_or_else(|e| die(&format!("cannot write {}: {}", spec.path, e)));
}

impl Tracer {
    pub fn load_csv(&mut self, path: &str) -> TVal {
        let table = read_table(path);
        let n = table.columns[0].len();
        let mut fields = Vec::new();
        for name in &table.names {
            let existing = self.inputs.iter().find(|(src, _)| {
                matches!(src, InputSource::Csv(p, c) if p == path && c == name)
            }).map(|&(_, id)| id);
            let val = match existing {
                Some(id) => self.val(id),
                None => {
                    let val = self.emit(OpKind::Input, vec![], vec![n], Dtype::F32);
                    self.inputs.push((InputSource::Csv(path.to_string(), name.clone()), val.id));
                    val
                }
            };
            fields.push((name.clone(), TVal::Tensor(BVal { val, bdims: 0 })));
        }
        TVal::Record(None, fields)
    }
}
