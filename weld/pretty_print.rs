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
            Builder(Appender(ref t), ref annotations) => {
                format!("{}appender[{}]", annotations, t.print())
            }
            Builder(DictMerger(ref kt, ref vt, op), ref annotations) => {
                format!("{}dictmerger[{},{},{}]",
                        annotations,
                        kt.print(),
                        vt.print(),
                        op)
            }
            Builder(GroupMerger(ref kt, ref vt), ref annotations) => {
                format!("{}groupmerger[{},{}]", annotations, kt.print(), vt.print())
            }
            Builder(VecMerger(ref elem, op), ref annotations) => {
                format!("{}vecmerger[{},{}]", annotations, elem.print(), op)
            }
            Builder(Merger(ref t, op), ref annotations) => {
                format!("{}merger[{},{}]", annotations, t.print(), op)
            }
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
            Builder(Appender(ref elem), ref annotations) => {
                format!("{}appender[{}]", annotations, elem.print())
            }
            Builder(DictMerger(ref kt, ref vt, _, op), ref annotations) => {
                format!("{}dictmerger[{},{},{}]",
                        annotations,
                        kt.print(),
                        vt.print(),
                        op)
            }
            Builder(VecMerger(ref elem, _, op), ref annotations) => {
                format!("{}vecmerger[{},{}]", annotations, elem.print(), op)
            }
            Builder(Merger(ref t, op), ref annotations) => {
                format!("{}merger[{},{}]", t.print(), annotations, op)
            }
            Builder(GroupMerger(ref kt, ref vt, _), ref annotations) => {
                format!("{}groupmerger[{},{}]", annotations, kt.print(), vt.print())
            }
        }
    }
}

/// Print a type.
pub fn print_type<T: PrintableType>(ty: &T) -> String {
    ty.print()
}

/// Print an expression concisely (without any type annotations).
pub fn print_expr<T: PrintableType>(expr: &Expr<T>) -> String {
    print_expr_impl(expr, false, 2, true)
}

/// Print an expression concisely (without any type annotations), without indentation.
pub fn print_expr_without_indent<T: PrintableType>(expr: &Expr<T>) -> String {
    print_expr_impl(expr, false, 2, false)
        .chars()
        .map(|x| match x {
                 '\n' => "".to_string(),
                 _ => x.to_string(),
             })
        .collect()
}

/// Print an expression with type annotations on all symbols.
pub fn print_typed_expr<T: PrintableType>(expr: &Expr<T>) -> String {
    print_expr_impl(expr, true, 2, true)
}

/// Print an expression with type annotations on all symbols, without indentation.
pub fn print_typed_expr_without_indent<T: PrintableType>(expr: &Expr<T>) -> String {
    print_expr_impl(expr, true, 2, false)
        .chars()
        .map(|x| match x {
                 '\n' => "".to_string(),
                 _ => x.to_string(),
             })
        .collect()
}

fn print_iters<T: PrintableType>(iters: &Vec<Iter<T>>,
                                 typed: bool,
                                 indent: i32,
                                 should_indent: bool)
                                 -> String {
    let mut iter_strs = vec![];
    let mut less_indent_str: String = iter::repeat(" ").take((indent - 2) as usize).collect();
    let mut indent_str: String = iter::repeat(" ").take(indent as usize).collect();
    if !should_indent {
        less_indent_str = "".to_string();
        indent_str = "".to_string();
    }
    for iter in iters {
        if let Some(_) = iter.start {
            iter_strs.push(format!("iter({},{},{},{})",
                                   print_expr_impl(iter.data.as_ref(),
                                                   typed,
                                                   indent,
                                                   should_indent),
                                   print_expr_impl(iter.start.as_ref().unwrap(),
                                                   typed,
                                                   indent,
                                                   should_indent),
                                   print_expr_impl(iter.end.as_ref().unwrap(),
                                                   typed,
                                                   indent,
                                                   should_indent),
                                   print_expr_impl(iter.stride.as_ref().unwrap(),
                                                   typed,
                                                   indent,
                                                   should_indent)));
        } else {
            iter_strs.push(print_expr_impl(iter.data.as_ref(), typed, indent + 2, should_indent));
        }
    }

    if iters.len() > 1 {
        format!("zip(\n{}{}\n{})",
                indent_str,
                iter_strs.join(&format!(",\n{}", indent_str)),
                less_indent_str)
    } else {
        iter_strs[0].clone()
    }
}

/// Main work to print an expression.
fn print_expr_impl<T: PrintableType>(expr: &Expr<T>,
                                     typed: bool,
                                     indent: i32,
                                     should_indent: bool)
                                     -> String {
    let mut less_indent_str: String = iter::repeat(" ").take((indent - 2) as usize).collect();
    let mut indent_str: String = iter::repeat(" ").take(indent as usize).collect();
    if !should_indent {
        less_indent_str = "".to_string();
        indent_str = "".to_string();
    }
    match expr.kind {
        Literal(ref lit) => print_literal(lit),

        Ident(ref symbol) => {
            if typed {
                format!("{}:{}", symbol, expr.ty.print())
            } else {
                format!("{}", symbol)
            }
        }

        BinOp {
            kind,
            ref left,
            ref right,
        } => {
            format!("({}{}{})",
                    print_expr_impl(left, typed, indent, should_indent),
                    kind,
                    print_expr_impl(right, typed, indent, should_indent))
        }

        Negate(ref e) => format!("(-{})", print_expr_impl(e, typed, indent, should_indent)),

        CUDF {
            ref sym_name,
            ref args,
            ref return_ty,
        } => {
            format!("cudf[{},{}]{}",
                    sym_name,
                    return_ty.print(),
                    join("(",
                         ",",
                         ")",
                         args.iter()
                             .map(|e| print_expr_impl(e, typed, indent, should_indent))))
        }

        Cast {
            kind,
            ref child_expr,
        } => {
            format!("({}({}))",
                    kind,
                    print_expr_impl(child_expr, typed, indent, should_indent))
        }

        ToVec { ref child_expr } => {
            format!("toVec({})",
                    print_expr_impl(child_expr, typed, indent, should_indent))
        }

        Let {
            ref name,
            ref value,
            ref body,
        } => {
            if typed {
                format!("(let {}:{}=({});{})",
                        name,
                        value.ty.print(),
                        print_expr_impl(value, typed, indent, should_indent),
                        print_expr_impl(body, typed, indent, should_indent))
            } else {
                format!("(let {}=({});{})",
                        name,
                        print_expr_impl(value, typed, indent, should_indent),
                        print_expr_impl(body, typed, indent, should_indent))
            }
        }

        MakeStruct { ref elems } => {
            join("{",
                 ",",
                 "}",
                 elems
                     .iter()
                     .map(|e| print_expr_impl(e, typed, indent, should_indent)))
        }

        MakeVector { ref elems } => {
            join("[",
                 ",",
                 "]",
                 elems
                     .iter()
                     .map(|e| print_expr_impl(e, typed, indent, should_indent)))
        }

        Zip { ref vectors } => {
            join(&format!("zip(\n{}", indent_str),
                 &format!(",\n{}", indent_str),
                 ")",
                 vectors
                     .iter()
                     .map(|e| print_expr_impl(e, typed, indent + 2, should_indent)))
        }

        GetField { ref expr, index } => {
            format!("{}.${}",
                    print_expr_impl(expr, typed, indent, should_indent),
                    index)
        }

        Length { ref data } => {
            format!("len({})",
                    print_expr_impl(data, typed, indent, should_indent))
        }
        Lookup {
            ref data,
            ref index,
        } => {
            format!("lookup({},{})",
                    print_expr_impl(data, typed, indent, should_indent),
                    print_expr_impl(index, typed, indent, should_indent))
        }
        KeyExists { ref data, ref key } => {
            format!("keyexists({},{})",
                    print_expr_impl(data, typed, indent, should_indent),
                    print_expr_impl(key, typed, indent, should_indent))
        }

        Slice {
            ref data,
            ref index,
            ref size,
        } => {
            format!("slice({},{},{})",
                    print_expr_impl(data, typed, indent, should_indent),
                    print_expr_impl(index, typed, indent, should_indent),
                    print_expr_impl(size, typed, indent, should_indent))
        }

        Exp { ref value } => {
            format!("exp({})",
                    print_expr_impl(value, typed, indent, should_indent))
        }

        Lambda {
            ref params,
            ref body,
        } => {
            let mut res = join("|",
                               ",",
                               "|",
                               params.iter().map(|e| print_parameter(e, typed)));
            res.push_str(&format!("\n{}{}",
                                 indent_str,
                                 print_expr_impl(body, typed, indent + 2, should_indent)));
            res
        }

        NewBuilder(ref arg) => {
            match *arg {
                Some(ref e) => {
                    format!("{}({})",
                            expr.ty.print(),
                            print_expr_impl(e, typed, indent, should_indent))
                }
                None => expr.ty.print(),
            }
        }

        Res { ref builder } => {
            format!("result(\n{}{}\n{})",
                    indent_str,
                    print_expr_impl(builder, typed, indent + 2, should_indent),
                    less_indent_str)
        }

        Merge {
            ref builder,
            ref value,
        } => {
            format!("merge({},{})",
                    print_expr_impl(builder, typed, indent, should_indent),
                    print_expr_impl(value, typed, indent, should_indent))
        }

        For {
            ref iters,
            ref builder,
            ref func,
        } => {
            format!("for(\n{}{},\n{}{},\n{}{}\n{})",
                    indent_str,
                    print_iters(iters, typed, indent + 2, should_indent),
                    indent_str,
                    print_expr_impl(builder, typed, indent + 2, should_indent),
                    indent_str,
                    print_expr_impl(func, typed, indent + 2, should_indent),
                    less_indent_str)
        }

        If {
            ref cond,
            ref on_true,
            ref on_false,
        } => {
            format!("if(\n{}{},\n{}{},\n{}{}\n{})",
                    indent_str,
                    print_expr_impl(cond, typed, indent + 2, should_indent),
                    indent_str,
                    print_expr_impl(on_true, typed, indent + 2, should_indent),
                    indent_str,
                    print_expr_impl(on_false, typed, indent + 2, should_indent),
                    less_indent_str)
        }

        Apply {
            ref func,
            ref params,
        } => {
            let mut res = format!("({})", print_expr_impl(func, typed, indent, should_indent));
            res.push_str(&join("(",
                               ",",
                               ")",
                               params
                                   .iter()
                                   .map(|e| print_expr_impl(e, typed, indent, should_indent))));
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
