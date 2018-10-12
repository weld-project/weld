//! Utilities used in unit testing.

use ast::Expr;

pub use syntax::parser::parse_expr;
pub use ast::CompareIgnoringSymbols;

/// Returns a typed expression.
#[cfg(test)]
pub fn typed_expression(s: &str) -> Expr {
    use ast::InferTypes;
    let mut expr = parse_expr(s).unwrap();
    expr.infer_types().unwrap();
    expr
}

/// Checks whether transform(input) == expect.
#[cfg(test)]
pub fn check_transform(input: &str,
                       expect: &str,
                       transform: fn(&mut Expr)) {

    use ast::{PrettyPrint, PrettyPrintConfig};

    let ref conf = PrettyPrintConfig::new()
        .show_types(true);

    let ref mut input = typed_expression(input);
    let ref expect = typed_expression(expect);

    transform(input);

    println!("\nInput: {}\nExpect: {}",
             input.pretty_print_config(conf),
             expect.pretty_print_config(conf));

    assert!(input.compare_ignoring_symbols(expect).unwrap());
}

/// Print an un-indented expression for string comparison.
#[cfg(test)]
pub fn print_expr_without_indent(e: &Expr) -> String {
    use ast::{PrettyPrintConfig, PrettyPrint};
    let ref config = PrettyPrintConfig::new()
                            .show_types(false)
                            .should_indent(false);
    e.pretty_print_config(config)
}

/// Print an un-indented expression for string comparison.
#[cfg(test)]
pub fn print_typed_expr_without_indent(e: &Expr) -> String {
    use ast::{PrettyPrintConfig, PrettyPrint};
    let ref config = PrettyPrintConfig::new()
                            .show_types(true)
                            .should_indent(false);
    e.pretty_print_config(config)
}
