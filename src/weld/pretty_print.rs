
use super::ast::BinOpKind;
use super::ast::BinOpKind::*;
use super::ast::ScalarKind::*;
use super::parser::*;
use super::parser::Type::*;
use super::parser::ExprKind::*;

// TODO: These methods could take a mutable string as an argument, or even a fmt::Format.

/// Print a data type.
pub fn print_type(ty: &Type) -> String {
    match *ty {
        Scalar(Bool) => "bool".to_string(),
        Scalar(I32) => "i32".to_string(),
        Vector(ref elem) => format!("vec[{}]", print_type(elem)),
        Function(ref params, ref ret) => {
            let mut res = String::new();
            res.push_str("(");
            for (i, p) in params.iter().enumerate() {
                if i > 0 { res.push_str(","); }
                res.push_str(&print_type(p));
            }
            res.push_str(")=>");
            res.push_str(&print_type(ret));
            res
        }
        _ => "?".to_string()
    }
}

/// Print a data type option, outputting "?" if unknown.
fn print_optional_type(ty: &Option<Type>) -> String {
    match *ty {
        Some(ref t) => print_type(t),
        None => "?".to_string()
    }
}

/// Print an expression concisely (without any type annotations).
pub fn print_expr(expr: &Expr) -> String {
    print_expr_impl(expr, false)
}

/// Print an expression with type annotations on all symbols.
pub fn print_typed_expr(expr: &Expr) -> String {
    print_expr_impl(expr, true)
}

/// Main work to print an expression.
fn print_expr_impl(expr: &Expr, typed: bool) -> String {
    match expr.kind {
        BoolLiteral(v) => format!("{}", v),
        I32Literal(v) => format!("{}", v),

        Ident(ref symbol) => {
            if typed {
                format!("{}:{}", symbol, print_optional_type(&expr.ty))
            } else {
                format!("{}", symbol)
            }
        }

        BinOp(kind, ref left, ref right) =>
            format!("({}{}{})",
                    print_expr_impl(left, typed),
                    print_binop(kind),
                    print_expr_impl(right, typed)),

        Let(ref symbol, ref value, ref body) => {
            if typed {
                format!("{}:{}:=({});{}",
                        symbol,
                        print_optional_type(&value.ty),
                        print_expr_impl(value, typed),
                        print_expr_impl(body, typed))
            } else {
                format!("{}:=({});{}",
                        symbol,
                        print_expr_impl(value, typed),
                        print_expr_impl(body, typed))
            }
        }

        MakeVector(ref exprs) => {
            let mut res = String::new();
            res.push_str("[");
            for (i, e) in exprs.iter().enumerate() {
                if i > 0 { res.push_str(","); }
                res.push_str(&print_expr_impl(e, typed));
            }
            res.push_str("]");
            res
        }

        Lambda(ref params, ref body) => {
            let mut res = String::new();
            res.push_str("|");
            for (i, p) in params.iter().enumerate() {
                if i > 0 { res.push_str(","); }
                res.push_str(&print_parameter(p, typed));
            }
            res.push_str("|");
            res.push_str(&print_expr_impl(body, typed));
            res
        }

        Map(ref data, ref body) =>
            format!("map({},{})",
                print_expr_impl(data, typed),
                print_expr_impl(body, typed)),

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

/// Print a parameter, optionally showing its type.
fn print_parameter(param: &Parameter, typed: bool) -> String {
    if typed {
        format!("{}:{}", param.name, print_optional_type(&param.ty))
    } else {
        format!("{}", param.name)
    }
}