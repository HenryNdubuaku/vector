use crate::graph::{Dtype, Node, OpKind, Val};

fn tensor_type(shape: &[usize], dtype: Dtype) -> String {
    let dims: String = shape.iter().map(|d| format!("{}x", d)).collect();
    format!("tensor<{}{}>", dims, dtype.name())
}

fn mlir_float(n: f64) -> String {
    if !n.is_finite() {
        return format!("0x{:016X}", n.to_bits());
    }
    let s = format!("{:?}", n);
    if s.contains('e') && !s.contains('.') {
        s.replace('e', ".0e")
    } else {
        s
    }
}

fn join(xs: &[usize]) -> String {
    xs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")
}

fn dense_text(vals: &[f64], shape: &[usize]) -> String {
    if shape.is_empty() {
        return mlir_float(vals[0]);
    }
    let inner: usize = shape[1..].iter().product::<usize>().max(1);
    let parts: Vec<String> = (0..shape[0])
        .map(|i| dense_text(&vals[i * inner..(i + 1) * inner], &shape[1..]))
        .collect();
    format!("[{}]", parts.join(", "))
}

fn val_name(id: usize, nodes: &[Node]) -> String {
    match &nodes[id].kind {
        OpKind::Barrier => val_name(nodes[id].inputs[0], nodes),
        OpKind::Proj(k) => {
            let w = nodes[id].inputs[0];
            let count = match &nodes[w].kind {
                OpKind::While { iter_args, .. } => iter_args.len(),
                _ => unreachable!("Proj of a non-while node"),
            };
            if count == 1 { format!("%{}", w) } else { format!("%{}#{}", w, k) }
        }
        _ => format!("%{}", id),
    }
}

fn node_text(node: &Node, nodes: &[Node]) -> String {
    let t = |i: usize| tensor_type(&nodes[node.inputs[i]].shape, nodes[node.inputs[i]].dtype);
    let arg = |i: usize| val_name(node.inputs[i], nodes);
    let out = tensor_type(&node.shape, node.dtype);
    match &node.kind {
        OpKind::Input => unreachable!("inputs are function parameters"),
        OpKind::IterArg => unreachable!("iter args are while binders"),
        OpKind::Proj(_) => unreachable!("projections are name aliases"),
        OpKind::Barrier => unreachable!("barriers are name aliases"),
        OpKind::While { .. } => unreachable!("while is emitted by the region writer"),
        OpKind::Iota => format!("stablehlo.iota dim = 0 : {}", out),
        OpKind::Constant(n) => {
            let lit = match node.dtype {
                Dtype::I64 => format!("{}", *n as i64),
                Dtype::F32 if !n.is_finite() => format!("0x{:08X}", (*n as f32).to_bits()),
                _ => mlir_float(*n),
            };
            format!("stablehlo.constant dense<{}> : {}", lit, out)
        }
        OpKind::DenseConst(vals) => format!("stablehlo.constant dense<{}> : {}", dense_text(vals, &node.shape), out),
        OpKind::Ewise(name) => format!("stablehlo.{} {}, {} : {}", name, arg(0), arg(1), out),
        OpKind::Unary(name) => format!("stablehlo.{} {} : {}", name, arg(0), out),
        OpKind::Convert => format!("stablehlo.convert {} : ({}) -> {}", arg(0), t(0), out),
        OpKind::Broadcast(dims) => format!(
            "stablehlo.broadcast_in_dim {}, dims = [{}] : ({}) -> {}",
            arg(0), join(dims), t(0), out
        ),
        OpKind::Reshape => format!("stablehlo.reshape {} : ({}) -> {}", arg(0), t(0), out),
        OpKind::Concat(dim) => {
            let operands: Vec<String> = (0..node.inputs.len()).map(arg).collect();
            let in_types: Vec<String> = (0..node.inputs.len()).map(t).collect();
            format!(
                "stablehlo.concatenate {}, dim = {} : ({}) -> {}",
                operands.join(", "), dim, in_types.join(", "), out
            )
        }
        OpKind::Reduce(reducer, axes) => format!(
            "stablehlo.reduce({} init: {}) applies stablehlo.{} across dimensions = [{}] : ({}, {}) -> {}",
            arg(0), arg(1), reducer, join(axes), t(0), t(1), out
        ),
        OpKind::Dot(lb, rb, lc, rc) => {
            let batching = if lb.is_empty() {
                String::new()
            } else {
                format!("batching_dims = [{}] x [{}], ", join(lb), join(rb))
            };
            format!(
                "stablehlo.dot_general {}, {}, {}contracting_dims = [{}] x [{}] : ({}, {}) -> {}",
                arg(0), arg(1), batching, join(lc), join(rc), t(0), t(1), out
            )
        }
        OpKind::Compare(dir) => format!(
            "stablehlo.compare {}, {}, {} : ({}, {}) -> {}",
            dir, arg(0), arg(1), t(0), t(1), out
        ),
        OpKind::Select => format!(
            "stablehlo.select {}, {}, {} : {}, {}",
            arg(0), arg(1), arg(2), t(0), out
        ),
        OpKind::DynSlice(sizes) => {
            let operands: Vec<String> = (0..node.inputs.len()).map(arg).collect();
            let in_types: Vec<String> = (0..node.inputs.len()).map(t).collect();
            format!(
                "stablehlo.dynamic_slice {}, sizes = [{}] : ({}) -> {}",
                operands.join(", "), join(sizes), in_types.join(", "), out
            )
        }
        OpKind::DynUpdateSlice => {
            let operands: Vec<String> = (0..node.inputs.len()).map(arg).collect();
            let in_types: Vec<String> = (0..node.inputs.len()).map(t).collect();
            format!(
                "stablehlo.dynamic_update_slice {} : ({}) -> {}",
                operands.join(", "), in_types.join(", "), out
            )
        }
        OpKind::Slice(dim, start, limit) => {
            let in_shape = &nodes[node.inputs[0]].shape;
            let ranges: Vec<String> = in_shape.iter().enumerate()
                .map(|(d, &e)| if d == *dim { format!("{}:{}", start, limit) } else { format!("0:{}", e) })
                .collect();
            format!("stablehlo.slice {} [{}] : ({}) -> {}", arg(0), ranges.join(", "), t(0), out)
        }
    }
}

fn write_while(s: &mut String, id: usize, nodes: &[Node], indent: usize) {
    let OpKind::While { iter_args, results, body, limit } = &nodes[id].kind else {
        unreachable!()
    };
    let ind = " ".repeat(indent);
    let binders: Vec<String> = iter_args.iter().zip(&nodes[id].inputs)
        .map(|(&a, &i)| format!("%{} = {}", a, val_name(i, nodes)))
        .collect();
    let types: Vec<String> = iter_args.iter()
        .map(|&a| tensor_type(&nodes[a].shape, nodes[a].dtype))
        .collect();
    let head = if iter_args.len() == 1 {
        format!("%{}", id)
    } else {
        format!("%{}:{}", id, iter_args.len())
    };
    s.push_str(&format!("{}{} = stablehlo.while({}) : {}\n", ind, head, binders.join(", "), types.join(", ")));
    s.push_str(&format!("{} cond {{\n", ind));
    let counter = tensor_type(&nodes[iter_args[0]].shape, nodes[iter_args[0]].dtype);
    s.push_str(&format!(
        "{}  %c{} = stablehlo.compare LT, %{}, {} : ({}, {}) -> tensor<i1>\n",
        ind, id, iter_args[0], val_name(*limit, nodes), counter, counter
    ));
    s.push_str(&format!("{}  stablehlo.return %c{} : tensor<i1>\n", ind, id));
    s.push_str(&format!("{} }} do {{\n", ind));
    write_region(s, body, nodes, indent + 2);
    let rnames: Vec<String> = results.iter().map(|&r| val_name(r, nodes)).collect();
    s.push_str(&format!("{}  stablehlo.return {} : {}\n", ind, rnames.join(", "), types.join(", ")));
    s.push_str(&format!("{} }}\n", ind));
}

fn write_region(s: &mut String, ids: &[usize], nodes: &[Node], indent: usize) {
    for &id in ids {
        match &nodes[id].kind {
            OpKind::Input | OpKind::IterArg | OpKind::Proj(_) | OpKind::Barrier => {}
            OpKind::While { .. } => write_while(s, id, nodes, indent),
            _ => {
                s.push_str(&" ".repeat(indent));
                s.push_str(&format!("%{} = {}\n", id, node_text(&nodes[id], nodes)));
            }
        }
    }
}

pub fn build_module(nodes: &[Node], params: &[usize], outputs: &[Val]) -> String {
    let types: Vec<String> = outputs.iter().map(|v| tensor_type(&v.shape, v.dtype)).collect();
    let names: Vec<String> = outputs.iter().map(|v| val_name(v.id, nodes)).collect();
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
    let params: Vec<String> = params.iter()
        .map(|&id| format!("%{}: {}", id, tensor_type(&nodes[id].shape, nodes[id].dtype)))
        .collect();
    let mut claimed = std::collections::HashSet::new();
    for node in nodes {
        if let OpKind::While { body, .. } = &node.kind {
            claimed.extend(body.iter().copied());
        }
    }
    let top: Vec<usize> = (0..nodes.len()).filter(|id| !claimed.contains(id)).collect();
    let mut s = String::new();
    s.push_str("module {\n");
    s.push_str(&format!("  func.func @main({}){} {{\n", params.join(", "), signature));
    write_region(&mut s, &top, nodes, 4);
    s.push_str(&ret);
    s.push_str("  }\n}\n");
    s
}
