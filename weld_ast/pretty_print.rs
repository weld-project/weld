use std::iter::Iterator;

use super::*;
use super::BinOpKind::*;
use super::BuilderKind::*;
use super::ScalarKind::*;
use super::Type::*;
use super::ExprKind::*;
use super::partial_types::*;

// TODO: These methods could take a mutable string as an argument, or even a fmt::Format.

/// A trait for printing types.
pub trait PrintableType: Clone {
    fn print(&self) -> String;
}

/// Print implementation for full Types
impl PrintableType for Type {
    fn print(&self) -> String {
        match *self {
            Scalar(Bool) => "bool".to_string(),
            Scalar(I32) => "i32".to_string(),
            Vector(ref elem) => format!("vec[{}]", elem.print()),
            Struct(ref elems) => join("{", ",", "}", elems.iter().map(|e| e.print())),
            Function(ref params, ref ret) => {
                let mut res = join("(", ",", ")=>", params.iter().map(|e| e.print()));
                res.push_str(&ret.print());
                res
            },
            Builder(Appender(ref t)) => format!("appender[{}]", t.print()),
            Builder(Merger(ref t, op)) => format!("merger[{},{}]", t.print(), print_binop(op)),
        }
    }
}

/// Print implementation for PartialTypes
impl PrintableType for PartialType {
    fn print(&self) -> String {
        use partial_types::PartialType::*;
        use partial_types::PartialBuilderKind::*;
        match *self {
            Unknown => "?".to_string(),
            Scalar(Bool) => "bool".to_string(),
            Scalar(I32) => "i32".to_string(),
            Vector(ref elem) => format!("vec[{}]", elem.print()),
            Struct(ref elems) => join("{", ",", "}", elems.iter().map(|e| e.print())),
            Function(ref params, ref ret) => {
                let mut res = join("(", ",", ")=>", params.iter().map(|e| e.print()));
                res.push_str(&ret.print());
                res
            },
            Builder(Appender(ref elem)) => format!("appender[{}]", elem.print()),
            Builder(Merger(ref t, op)) => format!("merger[{},{}]", t.print(), print_binop(op)),
        }
    }
}

/// Print a type.
pub fn print_type<T: PrintableType>(ty: &T) -> String {
    ty.print()
}

/// Print an expression concisely (without any type annotations).
pub fn print_expr<T: PrintableType>(expr: &Expr<T>) -> String {
    print_expr_impl(expr, false)
}

/// Print an expression with type annotations on all symbols.
pub fn print_typed_expr<T: PrintableType>(expr: &Expr<T>) -> String {
    print_expr_impl(expr, true)
}

/// Main work to print an expression.
fn print_expr_impl<T: PrintableType>(expr: &Expr<T>, typed: bool) -> String {
    match expr.kind {
        BoolLiteral(v) => format!("{}", v),
        I32Literal(v) => format!("{}", v),

        Ident(ref symbol) => {
            if typed {
                format!("{}:{}", symbol, expr.ty.print())
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
                format!("let {}:{}=({});{}",
                        symbol,
                        value.ty.print(),
                        print_expr_impl(value, typed),
                        print_expr_impl(body, typed))
            } else {
                format!("let {}=({});{}",
                        symbol,
                        print_expr_impl(value, typed),
                        print_expr_impl(body, typed))
            }
        }

        MakeStruct(ref exprs) =>
            join("{", ",", "}", exprs.iter().map(|e| print_expr_impl(e, typed))),

        MakeVector(ref exprs) =>
            join("[", ",", "]", exprs.iter().map(|e| print_expr_impl(e, typed))),

        GetField(ref param, index) => format!("{}.{}", print_expr_impl(param, typed), index),

        Lambda(ref params, ref body) => {
            let mut res = join("|", ",", "|", params.iter().map(|e| print_parameter(e, typed)));
            res.push_str(&print_expr_impl(body, typed));
            res
        }

        NewBuilder => expr.ty.print(),

        Res(ref builder) => format!("result({})", print_expr_impl(builder, typed)),

        Merge(ref builder, ref value) => {
            format!("merge({},{})", print_expr_impl(builder, typed), print_expr_impl(value, typed))
        }

        For(ref data, ref builder, ref func) => {
            format!("for({},{},{})",
                print_expr_impl(data, typed),
                print_expr_impl(builder, typed),
                print_expr_impl(func, typed))
        }

        If(ref cond, ref on_true, ref on_false) => {
            format!("if({},{},{})",
                print_expr_impl(cond, typed),
                print_expr_impl(on_true, typed),
                print_expr_impl(on_false, typed))
        }

        Apply(ref func, ref params) => {
            let mut res = format!("({})", print_expr_impl(func, typed));
            res.push_str(&join("(", ",", ")", params.iter().map(|e| print_expr_impl(e, typed))));
            res
        }
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
fn print_parameter<T: PrintableType>(param: &Parameter<T>, typed: bool) -> String {
    if typed {
        format!("{}:{}", param.name, param.ty.print())
    } else {
        format!("{}", param.name)
    }
}

fn join<T:Iterator<Item=String>>(start: &str, sep: &str, end: &str, strings: T) -> String {
    let mut res = String::new();
    res.push_str(start);
    for (i, s) in strings.enumerate() {
        if i > 0 { res.push_str(sep); }
        res.push_str(&s);
    }
    res.push_str(end);
    res
}