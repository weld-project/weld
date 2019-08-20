use crate::ast::ExprKind::*;
use crate::ast::*;

use crate::optimizer::transforms::vectorizer::ShouldPredicate;

#[cfg(test)]
use crate::tests::*;

pub fn short_circuit_booleans(expr: &mut Expr) {
    // For If statements annotated as predicated, do not apply the transform on the condition,
    // since it removes the predication opportunity.
    let should_predicate = expr.should_predicate();
    let applied = match expr.kind {
        If {
            ref mut on_true,
            ref mut on_false,
            ..
        } if should_predicate => {
            short_circuit_booleans(on_true);
            short_circuit_booleans(on_false);
            true
        }
        _ => false,
    };

    if applied {
        return;
    }

    let replaced = match expr.kind {
        BinOp {
            ref mut kind,
            ref mut left,
            ref mut right,
        } if *kind == BinOpKind::LogicalAnd => Some(
            Expr::new_if(
                *left.take(),
                *right.take(),
                Expr::new_literal(LiteralKind::BoolLiteral(false)).unwrap(),
            )
            .unwrap(),
        ),
        BinOp {
            ref mut kind,
            ref mut left,
            ref mut right,
        } if *kind == BinOpKind::LogicalOr => Some(
            Expr::new_if(
                *left.take(),
                Expr::new_literal(LiteralKind::BoolLiteral(true)).unwrap(),
                *right.take(),
            )
            .unwrap(),
        ),
        _ => None,
    };

    if let Some(replaced) = replaced {
        *expr = replaced;
    }

    for child in expr.children_mut() {
        short_circuit_booleans(child);
    }
}

/// Parse and perform type inference on an expression.
#[cfg(test)]
fn typed_expression(code: &str) -> Expr {
    let mut e = parse_expr(code).unwrap();
    assert!(e.infer_types().is_ok());
    e
}

#[test]
fn simple_and() {
    let mut e = typed_expression("|x:i32| (x > 5 && x < 10)");
    short_circuit_booleans(&mut e);
    let expect = &typed_expression("|x:i32| if (x > 5, x < 10, false)");
    assert!(e.compare_ignoring_symbols(expect).unwrap());
}

#[test]
fn simple_or() {
    let mut e = typed_expression("|x:i32| (x > 5 || x < 10)");
    short_circuit_booleans(&mut e);
    let expect = &typed_expression("|x:i32| if (x > 5, true, x < 10)");
    assert!(e.compare_ignoring_symbols(expect).unwrap());
}

#[test]
fn compound_and() {
    let mut e = typed_expression("|x:i32| x > 5 && x < 10 && x == 7");
    short_circuit_booleans(&mut e);
    let expect = &typed_expression("|x:i32| if ( if (x > 5, x < 10, false),  x == 7, false)");
    assert!(e.compare_ignoring_symbols(expect).unwrap());
}

#[test]
fn compound_or() {
    let mut e = typed_expression("|x:i32| (x > 5 || x < 10 || x == 7)");
    short_circuit_booleans(&mut e);
    let expect = &typed_expression("|x:i32| if ( if (x > 5, true, x < 10), true, x == 7 )");
    assert!(e.compare_ignoring_symbols(expect).unwrap());
}

#[test]
fn complex_and_or() {
    let mut e = typed_expression("|x:i32| (x > 5 || x < 10) && (x == 7 || x == 2)");
    short_circuit_booleans(&mut e);
    let expect = &typed_expression(
        "|x:i32|
                    if(
                        if (x > 5, true, x < 10),
                        if (x == 7, true, x == 2),
                        false  
                    )",
    );
    assert!(e.compare_ignoring_symbols(expect).unwrap());
}

#[test]
fn complex_and_or_2() {
    let mut e = typed_expression("|x:i32| (x > 5 && x < 10) || (x == 15)");
    short_circuit_booleans(&mut e);
    let expect = &typed_expression(
        "|x:i32|
                    if(
                        if (x > 5, x < 10, false),
                        true,
                        (x == 15)
                    )",
    );
    assert!(e.compare_ignoring_symbols(expect).unwrap());
}

#[test]
fn predicated_if() {
    let mut e =
        typed_expression("|x:i32| @(predicate:true) if (x > 5 && x < 10, x > 3 || x < 4, false)");
    short_circuit_booleans(&mut e);

    // Since the If is predicated, the condition should not be short-circuited.
    let expect = &typed_expression(
        "|x:i32|
                    @(predicate:true)
                    if (
                        x > 5 && x < 10, 
                        if (x > 3, true, x < 4),
                        false
                    )",
    );
    assert!(e.compare_ignoring_symbols(expect).unwrap());
}
