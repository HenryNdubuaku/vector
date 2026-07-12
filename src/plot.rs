use std::fs;

use crate::die;
use crate::graph::{Dtype, OpKind, TVal, Val};
use crate::runtime::Tensor;
use crate::trace::Tracer;

#[derive(Debug, Clone)]
pub struct SeriesSpec {
    pub scatter: bool,
    pub label: Option<String>,
    pub x: Val,
    pub y: Val,
}

#[derive(Debug, Clone, Default)]
pub struct FigureSpec {
    pub path: Option<String>,
    pub title: Option<String>,
    pub xlabel: Option<String>,
    pub ylabel: Option<String>,
    pub series: Vec<SeriesSpec>,
    pub images: Vec<Val>,
}

fn plot_vec(v: TVal, what: &str) -> Val {
    let b = v.tensor(what);
    if b.bdims != 0 {
        die(&format!("{} inside vmap isn't supported", what));
    }
    if b.val.dtype == Dtype::I1 {
        die("cannot plot booleans; use where to select values");
    }
    let s = &b.val.shape;
    let vector_like = s.len() == 1 || (s.len() == 2 && (s[0] == 1 || s[1] == 1));
    if !vector_like {
        die(&format!("{} expects vectors, got shape {:?}", what, s));
    }
    b.val
}

impl Tracer {
    pub fn plot_series(&mut self, scatter: bool, data: Vec<TVal>, label: Option<String>) -> TVal {
        let what = if scatter { "scatter" } else { "plot" };
        if self.region_depth > 0 {
            die(&format!("{} inside a for loop isn't supported (loops compile to one XLA while op); {} after the loop", what, what));
        }
        if !self.figure.images.is_empty() {
            die("imshow and plot in the same figure isn't supported");
        }
        let ret = data.last().unwrap().clone();
        let mut vals: Vec<Val> = data.into_iter().map(|v| plot_vec(v, what)).collect();
        let y = vals.pop().unwrap();
        let x = match vals.pop() {
            Some(x) => x,
            None => {
                let n: usize = y.shape.iter().product();
                self.emit(OpKind::Iota, vec![], vec![n], Dtype::F32)
            }
        };
        let nx: usize = x.shape.iter().product();
        let ny: usize = y.shape.iter().product();
        if nx != ny {
            die(&format!("{} x and y lengths differ: {} vs {}", what, nx, ny));
        }
        self.figure.series.push(SeriesSpec { scatter, label, x, y });
        ret
    }

    pub fn figure_text(&mut self, which: &str, s: String) {
        match which {
            "title" => self.figure.title = Some(s),
            "xlabel" => self.figure.xlabel = Some(s),
            _ => self.figure.ylabel = Some(s),
        }
    }

    pub fn imshow(&mut self, v: TVal) -> TVal {
        if self.region_depth > 0 {
            die("imshow inside a for loop isn't supported (loops compile to one XLA while op); imshow after the loop");
        }
        let b = match &v {
            TVal::Tensor(b) => b.clone(),
            TVal::Record(..) => die("imshow expects an image tensor"),
        };
        if b.bdims != 0 {
            die("imshow inside vmap isn't supported");
        }
        if b.val.dtype == Dtype::I1 {
            die("cannot show booleans; use where to select values");
        }
        crate::image::image_shape(&b.val.shape, "imshow");
        if !self.figure.series.is_empty() {
            die("imshow and plot in the same figure isn't supported");
        }
        if !self.figure.images.is_empty() {
            die("one imshow per figure; call savefig or show between images");
        }
        self.figure.images.push(b.val);
        v
    }

    pub fn finish_figure(&mut self, path: Option<String>) {
        if self.region_depth > 0 {
            die("savefig inside a for loop isn't supported (loops compile to one XLA while op); savefig after the loop");
        }
        if self.figure.series.is_empty() && self.figure.images.is_empty() {
            die("savefig without any plot; call plot, scatter or imshow first");
        }
        if let Some(p) = &path {
            if !p.ends_with(".svg") {
                die("savefig expects a path ending in .svg");
            }
            if self.figures.iter().any(|f| f.path.as_deref() == Some(p)) {
                die(&format!("duplicate savefig to {}", p));
            }
        }
        let mut fig = std::mem::take(&mut self.figure);
        fig.path = path;
        self.figures.push(fig);
    }
}

const PALETTE: [&str; 8] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
];

fn escape(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            c => out.push(c),
        }
    }
    out
}

fn nice_num(range: f64, round: bool) -> f64 {
    let exp = range.log10().floor();
    let f = range / 10f64.powf(exp);
    let nf = if round {
        if f < 1.5 { 1.0 } else if f < 3.0 { 2.0 } else if f < 7.0 { 5.0 } else { 10.0 }
    } else if f <= 1.0 { 1.0 } else if f <= 2.0 { 2.0 } else if f <= 5.0 { 5.0 } else { 10.0 };
    nf * 10f64.powf(exp)
}

fn ticks(lo: f64, hi: f64) -> (Vec<f64>, f64) {
    let range = nice_num(hi - lo, false);
    let step = nice_num(range / 4.0, true);
    let start = (lo / step).ceil() * step;
    let mut out = Vec::new();
    let mut v = start;
    while v <= hi + step * 1e-6 {
        out.push(v);
        v += step;
    }
    (out, step)
}

fn fmt_tick(v: f64, step: f64) -> String {
    let decimals = if step >= 1.0 { 0 } else { (-step.log10().floor()) as usize };
    let s = format!("{:.*}", decimals, v);
    if s == "-0" { "0".to_string() } else { s }
}

fn pad_range(lo: f64, hi: f64) -> (f64, f64) {
    if lo == hi {
        (lo - 1.0, hi + 1.0)
    } else {
        let pad = (hi - lo) * 0.05;
        (lo - pad, hi + pad)
    }
}

fn decorations(fig: &FigureSpec, s: &mut String, ml: f64, pw: f64, mt: f64, ph: f64, h: f64) {
    if let Some(t) = &fig.title {
        s.push_str(&format!(
            "<text x=\"{:.2}\" y=\"27\" font-size=\"15\" font-weight=\"600\" fill=\"#111\" text-anchor=\"middle\">{}</text>\n",
            ml + pw / 2.0, escape(t)
        ));
    }
    if let Some(t) = &fig.xlabel {
        s.push_str(&format!(
            "<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"13\" fill=\"#333\" text-anchor=\"middle\">{}</text>\n",
            ml + pw / 2.0, h - 12.0, escape(t)
        ));
    }
    if let Some(t) = &fig.ylabel {
        let cy = mt + ph / 2.0;
        s.push_str(&format!(
            "<text x=\"16\" y=\"{cy:.2}\" font-size=\"13\" fill=\"#333\" text-anchor=\"middle\" transform=\"rotate(-90 16 {cy:.2})\">{}</text>\n",
            escape(t)
        ));
    }
}

fn render_image(fig: &FigureSpec, t: &Tensor) -> String {
    let (w, h) = (640.0, 480.0);
    let (ml, mr, mt, mb) = (62.0, 22.0, 44.0, 50.0);
    let (pw, ph) = (w - ml - mr, h - mt - mb);
    let (ih, iw, c, pixels) = crate::image::image_bytes(t);
    let png = crate::image::encode_png(iw, ih, c, &pixels);
    let b64 = crate::image::base64(&png);
    let scale = (pw / iw as f64).min(ph / ih as f64);
    let (dw, dh) = (iw as f64 * scale, ih as f64 * scale);
    let (ix, iy) = (ml + (pw - dw) / 2.0, mt + (ph - dh) / 2.0);
    let mut s = String::new();
    s.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\" font-family=\"Helvetica, Arial, sans-serif\">\n"
    ));
    s.push_str(&format!("<rect width=\"{w}\" height=\"{h}\" fill=\"white\"/>\n"));
    s.push_str(&format!(
        "<image x=\"{ix:.2}\" y=\"{iy:.2}\" width=\"{dw:.2}\" height=\"{dh:.2}\" preserveAspectRatio=\"none\" style=\"image-rendering:pixelated\" href=\"data:image/png;base64,{b64}\"/>\n"
    ));
    s.push_str(&format!(
        "<rect x=\"{ix:.2}\" y=\"{iy:.2}\" width=\"{dw:.2}\" height=\"{dh:.2}\" fill=\"none\" stroke=\"#333\"/>\n"
    ));
    let (xticks, xstep) = ticks(0.0, iw as f64);
    for tk in &xticks {
        let px = ix + tk / iw as f64 * dw;
        s.push_str(&format!(
            "<line x1=\"{px:.2}\" y1=\"{:.2}\" x2=\"{px:.2}\" y2=\"{:.2}\" stroke=\"#333\"/>\n",
            iy + dh, iy + dh + 4.0
        ));
        s.push_str(&format!(
            "<text x=\"{px:.2}\" y=\"{:.2}\" font-size=\"11\" fill=\"#333\" text-anchor=\"middle\">{}</text>\n",
            iy + dh + 17.0, fmt_tick(*tk, xstep)
        ));
    }
    let (yticks, ystep) = ticks(0.0, ih as f64);
    for tk in &yticks {
        let py = iy + tk / ih as f64 * dh;
        s.push_str(&format!(
            "<line x1=\"{:.2}\" y1=\"{py:.2}\" x2=\"{ix:.2}\" y2=\"{py:.2}\" stroke=\"#333\"/>\n",
            ix - 4.0
        ));
        s.push_str(&format!(
            "<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"11\" fill=\"#333\" text-anchor=\"end\">{}</text>\n",
            ix - 8.0, py + 4.0, fmt_tick(*tk, ystep)
        ));
    }
    decorations(fig, &mut s, ml, pw, mt, ph, h);
    s.push_str("</svg>\n");
    s
}

fn render(fig: &FigureSpec, tensors: &[Tensor]) -> String {
    if !fig.images.is_empty() {
        return render_image(fig, &tensors[fig.series.len() * 2]);
    }
    let (w, h) = (640.0, 480.0);
    let (ml, mr, mt, mb) = (62.0, 22.0, 44.0, 50.0);
    let (pw, ph) = (w - ml - mr, h - mt - mb);
    let series: Vec<(Vec<f64>, Vec<f64>)> = (0..fig.series.len())
        .map(|i| {
            tensors[2 * i].f64_vec().iter()
                .zip(tensors[2 * i + 1].f64_vec())
                .filter(|(x, y)| x.is_finite() && y.is_finite())
                .map(|(&x, y)| (x, y))
                .unzip()
        })
        .collect();
    let points: Vec<(f64, f64)> = series.iter()
        .flat_map(|(xs, ys)| xs.iter().zip(ys).map(|(&x, &y)| (x, y)))
        .collect();
    if points.is_empty() {
        die("plot data has no finite values");
    }
    let (x0, x1) = pad_range(
        points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min),
        points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max),
    );
    let (y0, y1) = pad_range(
        points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min),
        points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max),
    );
    let sx = |v: f64| ml + (v - x0) / (x1 - x0) * pw;
    let sy = |v: f64| mt + ph - (v - y0) / (y1 - y0) * ph;
    let mut s = String::new();
    s.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\" font-family=\"Helvetica, Arial, sans-serif\">\n"
    ));
    s.push_str(&format!("<rect width=\"{w}\" height=\"{h}\" fill=\"white\"/>\n"));
    let (xticks, xstep) = ticks(x0, x1);
    for t in &xticks {
        let px = sx(*t);
        s.push_str(&format!(
            "<line x1=\"{px:.2}\" y1=\"{mt}\" x2=\"{px:.2}\" y2=\"{:.2}\" stroke=\"#e6e6e6\"/>\n",
            mt + ph
        ));
        s.push_str(&format!(
            "<text x=\"{px:.2}\" y=\"{:.2}\" font-size=\"11\" fill=\"#333\" text-anchor=\"middle\">{}</text>\n",
            mt + ph + 17.0, fmt_tick(*t, xstep)
        ));
    }
    let (yticks, ystep) = ticks(y0, y1);
    for t in &yticks {
        let py = sy(*t);
        s.push_str(&format!(
            "<line x1=\"{ml}\" y1=\"{py:.2}\" x2=\"{:.2}\" y2=\"{py:.2}\" stroke=\"#e6e6e6\"/>\n",
            ml + pw
        ));
        s.push_str(&format!(
            "<text x=\"{:.2}\" y=\"{:.2}\" font-size=\"11\" fill=\"#333\" text-anchor=\"end\">{}</text>\n",
            ml - 8.0, py + 4.0, fmt_tick(*t, ystep)
        ));
    }
    s.push_str(&format!(
        "<rect x=\"{ml}\" y=\"{mt}\" width=\"{pw}\" height=\"{ph}\" fill=\"none\" stroke=\"#333\"/>\n"
    ));
    for (i, ((xs, ys), spec)) in series.iter().zip(&fig.series).enumerate() {
        let color = PALETTE[i % PALETTE.len()];
        if spec.scatter {
            for (x, y) in xs.iter().zip(ys) {
                s.push_str(&format!(
                    "<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"3\" fill=\"{color}\"/>\n",
                    sx(*x), sy(*y)
                ));
            }
        } else {
            let pts: Vec<String> = xs.iter().zip(ys)
                .map(|(x, y)| format!("{:.2},{:.2}", sx(*x), sy(*y)))
                .collect();
            s.push_str(&format!(
                "<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"2\" points=\"{}\"/>\n",
                pts.join(" ")
            ));
        }
    }
    let labeled: Vec<(usize, &str)> = fig.series.iter().enumerate()
        .filter_map(|(i, spec)| spec.label.as_deref().map(|l| (i, l)))
        .collect();
    if !labeled.is_empty() {
        let widest = labeled.iter().map(|(_, l)| l.chars().count()).max().unwrap();
        let bw = widest as f64 * 7.2 + 44.0;
        let bh = labeled.len() as f64 * 18.0 + 10.0;
        let (bx, by) = (ml + pw - bw - 10.0, mt + 10.0);
        s.push_str(&format!(
            "<rect x=\"{bx:.2}\" y=\"{by:.2}\" width=\"{bw:.2}\" height=\"{bh:.2}\" rx=\"3\" fill=\"white\" stroke=\"#ccc\"/>\n"
        ));
        for (row, (i, label)) in labeled.iter().enumerate() {
            let cy = by + 14.0 + row as f64 * 18.0;
            let color = PALETTE[i % PALETTE.len()];
            if fig.series[*i].scatter {
                s.push_str(&format!(
                    "<circle cx=\"{:.2}\" cy=\"{:.2}\" r=\"3\" fill=\"{color}\"/>\n",
                    bx + 16.0, cy - 4.0
                ));
            } else {
                s.push_str(&format!(
                    "<line x1=\"{:.2}\" y1=\"{:.2}\" x2=\"{:.2}\" y2=\"{:.2}\" stroke=\"{color}\" stroke-width=\"2\"/>\n",
                    bx + 8.0, cy - 4.0, bx + 24.0, cy - 4.0
                ));
            }
            s.push_str(&format!(
                "<text x=\"{:.2}\" y=\"{cy:.2}\" font-size=\"12\" fill=\"#333\">{}</text>\n",
                bx + 30.0, escape(label)
            ));
        }
    }
    decorations(fig, &mut s, ml, pw, mt, ph, h);
    s.push_str("</svg>\n");
    s
}

pub fn write_figure(fig: &FigureSpec, tensors: &[Tensor], index: usize) -> String {
    let path = match &fig.path {
        Some(p) => p.clone(),
        None => std::env::temp_dir()
            .join(format!("vector_plot_{}.svg", index))
            .to_string_lossy()
            .into_owned(),
    };
    let svg = render(fig, tensors);
    fs::write(&path, svg).unwrap_or_else(|e| die(&format!("cannot write {}: {}", path, e)));
    path
}
