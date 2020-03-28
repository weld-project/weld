//! Utilities used in unit testing.

use crate::ast::Expr;

pub use crate::ast::CompareIgnoringSymbols;
pub use crate::syntax::parser::parse_expr;

/// Returns a typed expression.
#[cfg(test)]
pub fn typed_expression(s: &str) -> Expr {
    use crate::ast::InferTypes;
    let mut expr = parse_expr(s).unwrap();
    expr.infer_types().unwrap();
    expr
}

/// Checks whether transform(input) == expect.
#[cfg(test)]
pub fn check_transform(input: &str, expect: &str, transform: fn(&mut Expr)) {
    use crate::ast::{PrettyPrint, PrettyPrintConfig};

    let conf = PrettyPrintConfig::default().show_types(true);

    let input = &mut typed_expression(input);
    let expect = &typed_expression(expect);

    transform(input);
    assert!(input.compare_ignoring_symbols(expect).unwrap());
}

/// Print an un-indented expression for string comparison.
#[cfg(test)]
pub fn print_expr_without_indent(e: &Expr) -> String {
    use crate::ast::{PrettyPrint, PrettyPrintConfig};
    let config = PrettyPrintConfig::default()
        .show_types(false)
        .should_indent(false);
    e.pretty_print_config(&config)
}

/// Print an un-indented expression for string comparison.
#[cfg(test)]
pub fn print_typed_expr_without_indent(e: &Expr) -> String {
    use crate::ast::{PrettyPrint, PrettyPrintConfig};
    let config = PrettyPrintConfig::default()
        .show_types(true)
        .should_indent(false);
    e.pretty_print_config(&config)
}
