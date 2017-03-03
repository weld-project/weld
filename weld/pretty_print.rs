use std::iter;
use std::iter::Iterator;

use super::ast::*;
use super::ast::BuilderKind::*;
use super::ast::ScalarKind::*;
use super::ast::Type::*;
use super::ast::ExprKind::*;
use super::ast::LiteralKind::*;
use super::partial_types::*;

// TODO: These methods could take a mutable string as an argument, or even a fmt::Format.

/// A trait for printing types.
pub trait PrintableType: TypeBounds {
    fn print(&self) -> String;
}

/// Print implementation for full Types
impl PrintableType for Type {
    fn print(&self) -> String {
        match *self {
            Scalar(Bool) => "bool".to_string(),
            Scalar(I8) => "i8".to_string(),
            Scalar(I32) => "i32".to_string(),
            Scalar(I64) => "i64".to_string(),
            Scalar(F32) => "f32".to_string(),
            Scalar(F64) => "f64".to_string(),
            Vector(ref elem) => format!("vec[{}]", elem.print()),
            Dict(ref kt, ref vt) => format!("dict[{},{}]", kt.print(), vt.print()),
            Struct(ref elems) => join("{", ",", "}", elems.iter().map(|e| e.print())),
            Function(ref params, ref ret) => {
                let mut res = join("|", ",", "|(", params.iter().map(|e| e.print()));
                res.push_str(&ret.print());
                res.push_str(")");
                res
            }
            Builder(Appender(ref t)) => format!("appender[{}]", t.print()),
            Builder(DictMerger(ref kt, ref vt, op)) => {
                format!("dictmerger[{},{},{}]", kt.print(), vt.print(), op)
            }
            Builder(VecMerger(ref elem, op)) => format!("vecmerger[{},{}]", elem.print(), op),
            Builder(Merger(ref t, op)) => format!("merger[{},{}]", t.print(), op),
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
            Scalar(I8) => "i8".to_string(),
            Scalar(I32) => "i32".to_string(),
            Scalar(I64) => "i64".to_string(),
            Scalar(F32) => "f32".to_string(),
            Scalar(F64) => "f64".to_string(),
            Vector(ref elem) => format!("vec[{}]", elem.print()),
            Dict(ref kt, ref vt) => format!("dict[{},{}]", kt.print(), vt.print()),
            Struct(ref elems) => join("{", ",", "}", elems.iter().map(|e| e.print())),
            Function(ref params, ref ret) => {
                let mut res = join("(", ",", ")=>", params.iter().map(|e| e.print()));
                res.push_str(&ret.print());
                res
            }
            Builder(Appender(ref elem)) => format!("appender[{}]", elem.print()),
            Builder(DictMerger(ref kt, ref vt, _, op)) => {
                format!("dictmerger[{},{},{}]", kt.print(), vt.print(), op)
            }
            Builder(VecMerger(ref elem, _, op)) => format!("vecmerger[{},{}]", elem.print(), op),
            Builder(Merger(ref t, op)) => format!("merger[{},{}]", t.print(), op),
        }
    }
}

/// Print a type.
pub fn print_type<T: PrintableType>(ty: &T) -> String {
    ty.print()
}

/// Print an expression concisely (without any type annotations).
pub fn print_expr<T: PrintableType>(expr: &Expr<T>) -> String {
    print_expr_impl(expr, false, 2)
}

/// Print an expression with type annotations on all symbols.
pub fn print_typed_expr<T: PrintableType>(expr: &Expr<T>) -> String {
    print_expr_impl(expr, true, 2)
}

fn print_iters<T: PrintableType>(iters: &Vec<Iter<T>>, typed: bool, indent: i32) -> String {
    let mut iter_strs = vec![];
    let less_indent_str: String = iter::repeat(" ").take((indent - 2) as usize).collect();
    let indent_str: String = iter::repeat(" ").take(indent as usize).collect();
    for iter in iters {
        if let Some(_) = iter.start {
            iter_strs.push(format!("iter({},{},{},{})",
                                   print_expr_impl(iter.data.as_ref(), typed, indent),
                                   print_expr_impl(iter.start.as_ref().unwrap(), typed, indent),
                                   print_expr_impl(iter.end.as_ref().unwrap(), typed, indent),
                                   print_expr_impl(iter.stride.as_ref().unwrap(), typed, indent)));
        } else {
            iter_strs.push(print_expr_impl(iter.data.as_ref(), typed, indent + 2));
        }
    }

    if iters.len() > 1 {
        format!("zip(\n{}{}\n{})", indent_str, iter_strs.join(&format!(",\n{}", indent_str)), less_indent_str)
    } else {
        iter_strs[0].clone()
    }
}

/// Main work to print an expression.
fn print_expr_impl<T: PrintableType>(expr: &Expr<T>, typed: bool, indent: i32) -> String {
    let less_indent_str: String = iter::repeat(" ").take((indent - 2) as usize).collect();
    let indent_str: String = iter::repeat(" ").take(indent as usize).collect();
    match expr.kind {
        Literal(ref lit) => print_literal(lit),

        Ident(ref symbol) => {
            if typed {
                format!("{}:{}", symbol, expr.ty.print())
            } else {
                format!("{}", symbol)
            }
        }

        BinOp { kind, ref left, ref right } => {
            format!("({}{}{})",
                    print_expr_impl(left, typed, indent),
                    kind,
                    print_expr_impl(right, typed, indent))
        }

        Negate(ref e) => format!("(-{})", print_expr_impl(e, typed, indent)),

        Cast { kind, ref child_expr } => {
            format!("({}({}))", kind, print_expr_impl(child_expr, typed, indent))
        }

        ToVec { ref child_expr } => format!("toVec({})", print_expr_impl(child_expr, typed, indent)),

        Let { ref name, ref value, ref body } => {
            if typed {
                format!("(let {}:{}=({});{})",
                        name,
                        value.ty.print(),
                        print_expr_impl(value, typed, indent),
                        print_expr_impl(body, typed, indent))
            } else {
                format!("(let {}=({});{})",
                        name,
                        print_expr_impl(value, typed, indent),
                        print_expr_impl(body, typed, indent))
            }
        }

        MakeStruct { ref elems } => {
            join("{",
                 ",",
                 "}",
                 elems.iter().map(|e| print_expr_impl(e, typed, indent)))
        }

        MakeVector { ref elems } => {
            join("[",
                 ",",
                 "]",
                 elems.iter().map(|e| print_expr_impl(e, typed, indent)))
        }

        Zip { ref vectors } => {
            join(&format!("zip(\n{}", indent_str),
                 &format!(",\n{}", indent_str),
                 ")",
                 vectors.iter().map(|e| print_expr_impl(e, typed, indent + 2)))
        }

        GetField { ref expr, index } => format!("{}.${}", print_expr_impl(expr, typed, indent), index),

        Length { ref data } => format!("len({})", print_expr_impl(data, typed, indent)),
        Lookup { ref data, ref index } => {
            format!("lookup({},{})",
                    print_expr_impl(data, typed, indent),
                    print_expr_impl(index, typed, indent))
        }

        Slice { ref data, ref index, ref size } => {
            format!("slice({},{},{})",
                    print_expr_impl(data, typed, indent),
                    print_expr_impl(index, typed, indent),
                    print_expr_impl(size, typed, indent))
        }

        Exp { ref value } => format!("exp({})",print_expr_impl(value, typed, indent)),

        Lambda { ref params, ref body } => {
            let mut res = join("|",
                               ",",
                               "|",
                               params.iter().map(|e| print_parameter(e, typed)));
            res.push_str(&format!("\n{}{}", indent_str, print_expr_impl(body, typed, indent + 2)));
            res
        }

        NewBuilder(ref arg) => {
            match *arg {
                Some(ref e) => format!("{}({})", expr.ty.print(), print_expr_impl(e, typed, indent)),
                None => expr.ty.print(),
            }
        }

        Res { ref builder } => {
            format!("result(\n{}{}\n{})",
                    indent_str,
                    print_expr_impl(builder, typed, indent + 2),
                    less_indent_str)
        }

        Merge { ref builder, ref value } => {
            format!("merge({},{})",
                    print_expr_impl(builder, typed, indent),
                    print_expr_impl(value, typed, indent))
        }

        For { ref iters, ref builder, ref func } => {
            format!("for(\n{}{},\n{}{},\n{}{}\n{})",
                    indent_str,
                    print_iters(iters, typed, indent + 2),
                    indent_str,
                    print_expr_impl(builder, typed, indent + 2),
                    indent_str,
                    print_expr_impl(func, typed, indent + 2),
                    less_indent_str)
        }

        If { ref cond, ref on_true, ref on_false } => {
            format!("if(\n{}{},\n{}{},\n{}{}\n{})",
                    indent_str,
                    print_expr_impl(cond, typed, indent + 2),
                    indent_str,
                    print_expr_impl(on_true, typed, indent + 2),
                    indent_str,
                    print_expr_impl(on_false, typed, indent + 2),
                    less_indent_str)
        }

        Apply { ref func, ref params } => {
            let mut res = format!("({})", print_expr_impl(func, typed, indent));
            res.push_str(&join("(",
                               ",",
                               ")",
                               params.iter().map(|e| print_expr_impl(e, typed, indent))));
            res
        }
    }
}

/// Print a literal value.
pub fn print_literal(lit: &LiteralKind) -> String {
    match *lit {
        BoolLiteral(v) => format!("{}", v),
        I8Literal(v) => format!("{}", v),
        I32Literal(v) => format!("{}", v),
        I64Literal(v) => format!("{}L", v),
        F32Literal(v) => {
            let mut res = format!("{}", v);
            // Hack to disambiguate from integers.
            if !res.contains(".") {
                res.push_str(".0");
            }
            res.push_str("F");
            res
        }
        F64Literal(v) => {
            let mut res = format!("{}", v);
            // Hack to disambiguate from integers.
            if !res.contains(".") {
                res.push_str(".0");
            }
            res
        }
    }
}

/// Print a parameter, optionally showing its type.
fn print_parameter<T: PrintableType>(param: &Parameter<T>, typed: bool) -> String {
    if typed {
        format!("{}:{}", param.name, param.ty.print())
    } else {
        format!("{}", param.name)
    }
}

fn join<T: Iterator<Item = String>>(start: &str, sep: &str, end: &str, strings: T) -> String {
    let mut res = String::new();
    res.push_str(start);
    for (i, s) in strings.enumerate() {
        if i > 0 {
            res.push_str(sep);
        }
        res.push_str(&s);
    }
    res.push_str(end);
    res
}
