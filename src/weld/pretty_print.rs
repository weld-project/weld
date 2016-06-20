
use super::ast::BinOpKind;
use super::ast::BinOpKind::*;
use super::ast::ScalarKind::*;
use super::parser::*;
use super::parser::Type::*;
use super::parser::ExprKind::*;

// TODO: These methods could take a mutable string as an argument, or even a fmt::Format.

pub fn print_type(ty: &Type) -> String {
    match *ty {
        Scalar(Bool) => "bool".to_string(),
        Scalar(I32) => "i32".to_string(),
        Vector(ref elem) => format!("vec[{}]", print_type(elem)),
        _ => "?".to_string()
    }
}

fn print_optional_type(ty: &Option<Type>) -> String {
    match *ty {
        Some(ref t) => print_type(t),
        None => "?".to_string()
    }
}

pub fn print_expr(expr: &Expr) -> String {
    match expr.kind {
        BoolLiteral(v) => format!("{}", v),
        I32Literal(v) => format!("{}", v),
        Ident(ref symbol) => symbol.clone(),

        BinOp(kind, ref left, ref right) =>
            format!("({}{}{})", print_expr(left), print_binop(kind), print_expr(right)),

        Let(ref symbol, ref value, ref body) =>
            format!("{}:=({});{}", symbol, print_expr(value), print_expr(body)),

        _ => "<?>".to_string() 
    }
}

pub fn print_typed_expr(expr: &Expr) -> String {
    let t = print_optional_type(&expr.ty);
    match expr.kind {
        BoolLiteral(v) => format!("{}:{}", v, t),
        I32Literal(v) => format!("{}:{}", v, t),
        Ident(ref symbol) => format!("{}:{}", symbol, t),

        BinOp(kind, ref left, ref right) =>
            format!("({}{}{})",
                    print_typed_expr(left),
                    print_binop(kind),
                    print_typed_expr(right)),

        Let(ref symbol, ref value, ref body) =>
            format!("{}:{}:=({});{}",
                    symbol,
                    print_optional_type(&value.ty),
                    print_typed_expr(value),
                    print_typed_expr(body)),

        _ => "<?>".to_string() 
    }
}

fn print_binop(op: BinOpKind) -> String {
    match op {
        Add => "+",
        Subtract => "-",
        Multiply => "*",
        Divide => "/"
    }.to_string()
}
