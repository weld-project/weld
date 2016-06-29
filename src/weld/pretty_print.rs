
use super::ast::BinOpKind;
use super::ast::BinOpKind::*;
use super::ast::BuilderKind::*;
use super::ast::ScalarKind::*;
use super::ast::Expr;
use super::ast::Type;
use super::ast::Type::*;
use super::ast::Parameter;
use super::ast::ExprKind::*;
use super::partial_types::*;

// TODO: These methods could take a mutable string as an argument, or even a fmt::Format.

/// A trait for printing types.
pub trait PrintableType {
    fn print(&self) -> String;
}

/// Print implementation for full Types
impl PrintableType for Type {
    fn print(&self) -> String {
        match *self {
            Scalar(Bool) => "bool".to_string(),
            Scalar(I32) => "i32".to_string(),
            Vector(ref elem) => format!("vec[{}]", elem.print()),
            Function(ref params, ref ret) => {
                let mut res = String::new();
                res.push_str("(");
                for (i, p) in params.iter().enumerate() {
                    if i > 0 { res.push_str(","); }
                    res.push_str(&p.print());
                }
                res.push_str(")=>");
                res.push_str(&ret.print());
                res
            },
            Builder(Appender(ref elem)) => format!("appender[{}]", elem.print()),
            _ => "??".to_string()
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
            Function(ref params, ref ret) => {
                let mut res = String::new();
                res.push_str("(");
                for (i, p) in params.iter().enumerate() {
                    if i > 0 { res.push_str(","); }
                    res.push_str(&p.print());
                }
                res.push_str(")=>");
                res.push_str(&ret.print());
                res
            },
            Builder(Appender(ref elem)) => format!("appender[{}]", elem.print()),
            _ => "??".to_string()
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

        Map(ref data, ref body) => {
            format!("map({},{})", print_expr_impl(data, typed), print_expr_impl(body, typed))
        }

        NewBuilder => expr.ty.print(),

        Merge(ref builder, ref value) => {
            format!("merge({},{})", print_expr_impl(builder, typed), print_expr_impl(value, typed))
        }

        For(ref data, ref builder, ref func) => {
            format!("for({},{},{})",
                print_expr_impl(data, typed),
                print_expr_impl(builder, typed),
                print_expr_impl(func, typed))
        }

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
fn print_parameter<T: PrintableType>(param: &Parameter<T>, typed: bool) -> String {
    if typed {
        format!("{}:{}", param.name, param.ty.print())
    } else {
        format!("{}", param.name)
    }
}