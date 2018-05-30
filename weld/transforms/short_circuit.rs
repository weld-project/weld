
use ast::*;
use ast::ExprKind::*;
use exprs::*;

pub fn short_circuit_booleans(expr: &mut Expr) {
    // For If statements annotated as predicated, do not apply the transform on the condition,
    // since it removes the predication opportunity.
    let applied = match expr.kind {
        If { ref mut on_true, ref mut on_false, .. } if expr.annotations.predicate() => {
            short_circuit_booleans(on_true);
            short_circuit_booleans(on_false);
            true
        },
        _ => false,
    };

    if applied {
        return;
    }

    let new = match expr.kind {
        BinOp { ref kind, ref left, ref right } if *kind == BinOpKind::LogicalAnd => {
            Some(if_expr(left.as_ref().clone(), right.as_ref().clone(), literal_expr(LiteralKind::BoolLiteral(false)).unwrap()).unwrap())
        },
        BinOp { ref kind, ref left, ref right } if *kind == BinOpKind::LogicalOr => {
            Some(if_expr(left.as_ref().clone(), literal_expr(LiteralKind::BoolLiteral(true)).unwrap(), right.as_ref().clone()).unwrap())
        },
        _ => None,
    };

    if let Some(new) = new {
        *expr = new;
    }

    for child in expr.children_mut() {
        short_circuit_booleans(child);
    }
}

#[cfg(test)]
use parser::*;
#[cfg(test)]
use type_inference::*;

/// Parse and perform type inference on an expression.
#[cfg(test)]
fn typed_expr(code: &str) -> Expr {
    let mut e = parse_expr(code).unwrap();
    assert!(e.infer_types().is_ok());
    e
}

#[test]
fn simple_and() {
    let mut e = typed_expr("|x:i32| (x > 5 && x < 10)");
    short_circuit_booleans(&mut e);
    let ref expect = typed_expr("|x:i32| if (x > 5, x < 10, false)");
    assert!(e.compare_ignoring_symbols(expect).unwrap());
}


#[test]
fn simple_or() {
    let mut e = typed_expr("|x:i32| (x > 5 || x < 10)");
    short_circuit_booleans(&mut e);
    let ref expect = typed_expr("|x:i32| if (x > 5, true, x < 10)");
    assert!(e.compare_ignoring_symbols(expect).unwrap());
}

#[test]
fn compound_and() {
    let mut e = typed_expr("|x:i32| x > 5 && x < 10 && x == 7");
    short_circuit_booleans(&mut e);
    let ref expect = typed_expr("|x:i32| if ( if (x > 5, x < 10, false),  x == 7, false)");
    assert!(e.compare_ignoring_symbols(expect).unwrap());
}

#[test]
fn compound_or() {
    let mut e = typed_expr("|x:i32| (x > 5 || x < 10 || x == 7)");
    short_circuit_booleans(&mut e);
    let ref expect = typed_expr("|x:i32| if ( if (x > 5, true, x < 10), true, x == 7 )");
    assert!(e.compare_ignoring_symbols(expect).unwrap());
}

#[test]
fn complex_and_or() {
    let mut e = typed_expr("|x:i32| (x > 5 || x < 10) && (x == 7 || x == 2)");
    short_circuit_booleans(&mut e);
    let ref expect = typed_expr("|x:i32|
                    if(
                        if (x > 5, true, x < 10),
                        if (x == 7, true, x == 2),
                        false  
                    )");
    assert!(e.compare_ignoring_symbols(expect).unwrap());
}

#[test]
fn complex_and_or_2() {
    let mut e = typed_expr("|x:i32| (x > 5 && x < 10) || (x == 15)");
    short_circuit_booleans(&mut e);
    let ref expect = typed_expr("|x:i32|
                    if(
                        if (x > 5, x < 10, false),
                        true,
                        (x == 15)
                    )");
    assert!(e.compare_ignoring_symbols(expect).unwrap());
}

#[test]
fn predicated_if() {
    let mut e = typed_expr("|x:i32| @(predicate:true) if (x > 5 && x < 10, x > 3 || x < 4, false)");
    short_circuit_booleans(&mut e);

    // Since the If is predicated, the condition should not be short-circuited.
    let ref expect = typed_expr("|x:i32|
                    @(predicate:true)
                    if (
                        x > 5 && x < 10, 
                        if (x > 3, true, x < 4),
                        false
                    )");
    assert!(e.compare_ignoring_symbols(expect).unwrap());
}
