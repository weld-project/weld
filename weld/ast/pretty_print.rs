//! Pretty print Weld expression trees.
//!
//! This module prints Weld expressions. The printed expression is a valid Weld program and can be
//! re-parsed using the Weld API.

use ast::*;
use ast::ExprKind::*;

use util::join;

/// The number of spaces used when indenting code.
const INDENT_LEVEL: i32 = 2;

/// A trait for pretty printing expression trees.
pub trait PrettyPrint {
    /// Pretty print an expression.
    ///
    /// The pretty printed expression contains newlines and is indented to be human-readable.  The
    /// printed expression is a valid, parseable Weld program that can be re-parsed via the Weld
    /// API.
    fn pretty_print(&self) -> String;

    /// Pretty print an expression with the given configuration.
    ///
    /// The printed expression is a valid, parseable Weld program that can be
    /// re-parsed via the Weld API.
    fn pretty_print_config(&self, config: &PrettyPrintConfig) -> String;
}

impl PrettyPrint for Expr {
    /// Pretty print an expression.
    fn pretty_print(&self) -> String {
        let ref mut config = PrettyPrintConfig::new()
            .should_indent(true)
            .show_types(false);
        to_string_impl(self, config)
    }

    /// Pretty print an expression with the given configuration.
    fn pretty_print_config(&self, config: &PrettyPrintConfig) -> String {
        let ref mut config = config.clone();
        config.indent = INDENT_LEVEL;
        to_string_impl(self, config)
    }
}

/// A struct used to configure pretty printing.
///
/// This struct is used with the `PrettyPrint::pretty_print_config` function to configure how the
/// Weld expression is printed.
#[derive(Clone)]
pub struct PrettyPrintConfig {
    /// Sets whether to print the type of each symbol.
    pub  show_types: bool,
    /// Specifies whether to indent code.
    pub should_indent: bool,
    /// Tracks the indentation level.
    indent: i32,
    /// Specifies whether this is the top of the expression tree.
    top: bool,
}

impl PrettyPrintConfig {
    /// Returns a new default `PrettyPrintConfig`.
    ///
    /// The default configuration does not show types for each symbol name and indents code.
    pub fn new() -> Self {
        PrettyPrintConfig {
            show_types: false,
            should_indent: true,
            indent: INDENT_LEVEL,
            top: true,
        }
    }

    /// Set whether this print configuration should show types for each symbol.
    pub fn show_types(mut self, show_types: bool) -> PrettyPrintConfig {
        self.show_types = show_types;
        self
    }

    /// Set whether this print configuration should indent code.
    pub fn should_indent(mut self, should_indent: bool) -> PrettyPrintConfig {
        self.should_indent = should_indent;
        self
    }
}

/// Returns strings used for indentation.
///
/// Specifically, this function returns three strings: one representating the indented level of the
/// current configuration, one to represent one indent level lower, and one representing a newline
/// character.
fn indentation(config: &PrettyPrintConfig) -> (String, String, String) {
    if config.should_indent {
        (" ".repeat((config.indent - 2) as usize), " ".repeat(config.indent as usize), "\n".to_string())
    } else {
        ("".to_string(), "".to_string(), "".to_string())
    }
}

/// Prints a list of iterators for a For loop.
fn print_iters(iters: &Vec<Iter>, config: &mut PrettyPrintConfig) -> String {
    use self::IterKind::*;
    let mut iter_strs = vec![];
    let (less_indent_str, indent_str, newline) = indentation(config);
    for iter in iters {
        match iter.kind {
            NdIter => {
                /* Need to check this first because NdIter also has iter.start */
                iter_strs.push(format!("{}({},{},{},{})",
                    iter.kind,
                    to_string_impl(iter.data.as_ref(), config),
                    to_string_impl(iter.start.as_ref().unwrap(), config),
                    to_string_impl(iter.shape.as_ref().unwrap(), config),
                    to_string_impl(iter.strides.as_ref().unwrap(), config),
                ));
            }
            _ if iter.start.is_some() => {
                iter_strs.push(format!("{}({},{},{},{})",
                    iter.kind,
                    to_string_impl(iter.data.as_ref(), config),
                    to_string_impl(iter.start.as_ref().unwrap(), config),
                    to_string_impl(iter.end.as_ref().unwrap(), config),
                    to_string_impl(iter.stride.as_ref().unwrap(), config)
                ));
            }
            ScalarIter => {
                // don't print the Iter for ScalarIter for conciseness.
                iter_strs.push(to_string_impl(iter.data.as_ref(), config));
            }
            _ => {
                iter_strs.push(format!("{}({})", iter.kind, to_string_impl(iter.data.as_ref(), config)));
            }
        }
    }

    if iters.len() > 1 {
        format!("zip({}{}{}{}{})",
                newline,
                iter_strs.join(&format!(",{}{}", newline, indent_str)),
                indent_str,
                newline,
                less_indent_str)
    } else {
        iter_strs[0].clone()
    }
}

/// Pretty print each expression in an AST recursively.
fn to_string_impl(expr: &Expr, config: &mut PrettyPrintConfig) -> String {
    let (less_indent_str, indent_str, newline) = indentation(config);
    let top = config.top;
    config.top = false;
    let expr_str = match expr.kind {
        Literal(ref lit) => {
            lit.to_string()
        }

        Ident(ref symbol) if config.show_types => {
            format!("{}:{}", symbol, expr.ty)
        }

        Ident(ref symbol) => {
            symbol.to_string()
        }

        BinOp { kind, ref left, ref right } => {
            use super::ast::BinOpKind::*;
            let left = to_string_impl(left, config);
            let right = to_string_impl(right, config);
            match kind {
                Max | Min | Pow => format!("{}({},{})", kind, left, right),
                _ => format!("({}{}{})", left, kind, right)
            }
        }

        UnaryOp { kind, ref value } => {
            format!("({}({}))", kind, to_string_impl(value, config))
        }

        Negate(ref e) => {
            format!("(-{})", to_string_impl(e, config))
        }

        Not(ref e) => {
            format!("(!{})", to_string_impl(e, config))
        }

        Assert(ref e) => {
            format!("assert({})", to_string_impl(e, config))
        }

        Broadcast(ref e) => {
            format!("broadcast({})", to_string_impl(e, config))
        }

        CUDF { ref sym_name, ref args, ref return_ty } => {
            let args = join("(", ",", ")", args.iter().map(|e| to_string_impl(e, config)));
            format!("cudf[{},{}]{}", sym_name, return_ty, args)
        }

        Serialize(ref e) => {
            format!("serialize({})", to_string_impl(e, config))
        }

        Deserialize { ref value, ref value_ty } => {
            format!("deserialize[{}]({})", value_ty, to_string_impl(value, config))
        }

        Cast { kind, ref child_expr } => {
            format!("({}({}))", kind, to_string_impl(child_expr, config))
        }

        ToVec { ref child_expr } => {
            format!("tovec({})", to_string_impl(child_expr, config))
        }

        Let { ref name, ref value, ref body } => {
            let value_str = to_string_impl(value, config);
            let body = to_string_impl(body, config);
            if config.show_types {
                format!("(let {}:{}=({});{}{}{})", name, value.ty, value_str, newline, indent_str, body)
            } else {
                format!("(let {}=({});{}{}{})", name, value_str, newline, indent_str, body)
            }
        }

        MakeStruct { ref elems } => {
            join("{", ",", "}", elems.iter().map(|e| to_string_impl(e, config)))
        }

        MakeVector { ref elems } => {
            join("[", ",", "]", elems.iter().map(|e| to_string_impl(e, config)))
        }

        Zip { ref vectors } => {
            let ref begin = format!("zip({}{}", newline, indent_str);
            let ref newlines = format!(",{}{}", newline, indent_str);
            join(begin, newlines, ")", vectors.iter().
                map(&mut |e| {
                    config.indent += INDENT_LEVEL;
                    let result = to_string_impl(e, config);
                    config.indent -= INDENT_LEVEL;
                    result
            }))
        }

        GetField { ref expr, index } => {
            // If the expression is a symbol, don't show its type in a GetField.
            if config.show_types {
                if let Ident(ref symbol) = expr.kind {
                    return format!("{}.${}", symbol, index)
                }
            }
            format!("{}.${}", to_string_impl(expr, config), index)
        }

        Length { ref data } => {
            format!("len({})", to_string_impl(data, config))
        }

        Lookup { ref data, ref index, } => {
            format!("lookup({},{})", to_string_impl(data, config), to_string_impl(index, config))
        }

        OptLookup { ref data, ref index, } => {
            format!("optlookup({},{})", to_string_impl(data, config), to_string_impl(index, config))
        }

        KeyExists { ref data, ref key } => {
            format!("keyexists({},{})", to_string_impl(data, config), to_string_impl(key, config))
        }

        Slice { ref data, ref index, ref size } => {
            format!("slice({},{},{})",
                    to_string_impl(data, config),
                    to_string_impl(index, config),
                    to_string_impl(size, config))
        }

        Sort { ref data, ref cmpfunc } => {
            format!("sort({},{})",
                    to_string_impl(data, config),
                    to_string_impl(cmpfunc, config))
        }

        Lambda { ref params, ref body } => {
            // Print types for a top-level Lambda even if show_types is disabled - this allows
            // type inference to infer the remaining types in the program.
            let mut res = join("|", ",", "|", params.iter()
                               .map(|e| if top || config.show_types {
                                   e.to_string()
                               } else {
                                   e.name.to_string()
                               }));
            config.indent += INDENT_LEVEL;
            let body = to_string_impl(body, config);
            config.indent -= INDENT_LEVEL;
            res.push_str(&format!("{}{}{}", newline, indent_str, body));
            res
        }

        NewBuilder(ref arg) => {
            match *arg {
                Some(ref e) => format!("{}({})", expr.ty, to_string_impl(e, config)),
                None => expr.ty.to_string(),
            }
        }

        Res { ref builder } => {
            config.indent += INDENT_LEVEL;
            let builder = to_string_impl(builder, config);
            config.indent -= INDENT_LEVEL;
            format!("result({}{}{}{}{})", newline, indent_str, builder, newline, less_indent_str)
        }

        Merge { ref builder, ref value } => {
            format!("merge({},{})", to_string_impl(builder, config), to_string_impl(value, config))
        }

        For { ref iters, ref builder, ref func } => {
            config.indent += INDENT_LEVEL;
            let res = format!("for({}{}{},{}{}{},{}{}{}{}{})",
                    newline, indent_str, print_iters(iters, config),
                    newline, indent_str, to_string_impl(builder, config),
                    newline, indent_str, to_string_impl(func, config),
                    newline, less_indent_str);
            config.indent -= INDENT_LEVEL;
            res
        }

        If { ref cond, ref on_true, ref on_false } => {
            config.indent += INDENT_LEVEL;
            let res = format!("if({}{}{},{}{}{},{}{}{}{}{})",
                    newline, indent_str, to_string_impl(cond, config),
                    newline, indent_str, to_string_impl(on_true, config),
                    newline, indent_str, to_string_impl(on_false, config),
                    newline, less_indent_str);
            config.indent -= INDENT_LEVEL;
            res
        }

        Iterate { ref initial, ref update_func } => {
            config.indent += INDENT_LEVEL;
            let res = format!("iterate({}{}{},{}{}{}{}{})",
                    newline, indent_str, to_string_impl(initial, config),
                    newline, indent_str, to_string_impl(update_func, config),
                    newline, less_indent_str);
            config.indent -= INDENT_LEVEL;
            res
        }

        Select { ref cond, ref on_true, ref on_false } => {
            config.indent += INDENT_LEVEL;
            let res = format!("select({}{}{},{}{}{},{}{}{}{}{})",
                    newline, indent_str, to_string_impl(cond, config),
                    newline, indent_str, to_string_impl(on_true, config),
                    newline, indent_str, to_string_impl(on_false, config),
                    newline, less_indent_str);
            config.indent -= INDENT_LEVEL;
            res
        }

        Apply { ref func, ref params } => {
            let mut res = format!("({})", to_string_impl(func, config));
            let params = params.iter().map(|e| to_string_impl(e, config));
            res.push_str(&join("(", ",", ")", params));
            res
        }
    };
    format!("{}{}", expr.annotations, expr_str)
}
