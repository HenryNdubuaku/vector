use crate::graph::{Dtype, Node, OpKind, Val};
use crate::trace::Tracer;

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

fn join(xs: &[usize]) -> String {
    xs.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")
}

fn node_text(node: &Node, nodes: &[Node]) -> String {
    let t = |i: usize| tensor_type(&nodes[node.inputs[i]].shape, nodes[node.inputs[i]].dtype);
    let arg = |i: usize| format!("%{}", node.inputs[i]);
    let out = tensor_type(&node.shape, node.dtype);
    match &node.kind {
        OpKind::Input => unreachable!("inputs are function parameters"),
        OpKind::Constant(n) => format!("stablehlo.constant dense<{}> : {}", mlir_float(*n), out),
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
        OpKind::Reduce(axes) => format!(
            "stablehlo.reduce({} init: {}) applies stablehlo.add across dimensions = [{}] : ({}, {}) -> {}",
            arg(0), arg(1), join(axes), t(0), t(1), out
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
        OpKind::Slice(dim, start, limit) => {
            let in_shape = &nodes[node.inputs[0]].shape;
            let ranges: Vec<String> = in_shape.iter().enumerate()
                .map(|(d, &e)| if d == *dim { format!("{}:{}", start, limit) } else { format!("0:{}", e) })
                .collect();
            format!("stablehlo.slice {} [{}] : ({}) -> {}", arg(0), ranges.join(", "), t(0), out)
        }
    }
}

pub fn build_module(tracer: &Tracer, outputs: &[Val]) -> String {
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
    let params: Vec<String> = tracer.inputs.iter()
        .map(|&(_, id)| format!("%{}: {}", id, tensor_type(&tracer.nodes[id].shape, tracer.nodes[id].dtype)))
        .collect();
    let mut s = String::new();
    s.push_str("module {\n");
    s.push_str(&format!("  func.func @main({}){} {{\n", params.join(", "), signature));
    for (id, node) in tracer.nodes.iter().enumerate() {
        if matches!(node.kind, OpKind::Input) {
            continue;
        }
        s.push_str(&format!("    %{} = {}\n", id, node_text(node, &tracer.nodes)));
    }
    s.push_str(&ret);
    s.push_str("  }\n}\n");
    s
}
