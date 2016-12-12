use super::ast::{Expr, Type, ExprKind, Symbol};
use super::partial_types::PartialType::Unknown;
use super::parser::parse_expr;
use super::pretty_print::*;
use super::type_inference::*;

use super::transforms::fuse_loops;

/// Returns a typed expression.
#[cfg(test)]
fn typed_expression(s: &str) -> Expr<Type> {
    let mut e1 = parse_expr(s).unwrap();
    infer_types(&mut e1).unwrap();
    e1.to_typed().unwrap()
}

#[test]
fn parse_and_print_literal_expressions() {
    let tests = vec![
        // i32 literal expressions
        ("23", "23"),
        ("0b111", "7"),
        ("0xff", "255"),
        // i64 literal expressions
        ("23L", "23L"),
        ("7L", "7L"),
        ("0xffL", "255L"),
        // f64 literal expressions
        ("23.0", "23.0"),
        ("23.5", "23.5"),
        ("23e5", "2300000.0"),
        ("23.5e5", "2350000.0"),
        // f32 literal expressions
        ("23.0f", "23.0F"),
        ("23.5f", "23.5F"),
        ("23e5f", "2300000.0F"),
        ("23.5e5f", "2350000.0F"),
        // bool literal expressions
        ("true", "true"),
        ("false", "false"),
    ];

    for test in tests {
        let e = parse_expr(test.0).unwrap();
        assert_eq!(print_expr(&e).as_str(), test.1);
    }

    // Test overflow of integer types
    assert!(parse_expr("999999999999999").is_err());  // i32 literal too big
    assert!(parse_expr("999999999999999L").is_ok());
    assert!(parse_expr("999999999999999999999999999999L").is_err());  // i64 literal too big
}

#[test]
fn parse_and_print_simple_expressions() {
    let e = parse_expr("23 + 32").unwrap();
    assert_eq!(print_expr(&e).as_str(), "(23+32)");

    let e = parse_expr("2 - 3 - 4").unwrap();
    assert_eq!(print_expr(&e).as_str(), "((2-3)-4)");

    let e = parse_expr("2 - (3 - 4)").unwrap();
    assert_eq!(print_expr(&e).as_str(), "(2-(3-4))");

    let e = parse_expr("a").unwrap();
    assert_eq!(print_expr(&e).as_str(), "a");

    let e = parse_expr("let a = 2; a").unwrap();
    assert_eq!(print_expr(&e).as_str(), "(let a=(2);a)");

    let e = parse_expr("let a = 2.0; a").unwrap();
    assert_eq!(print_expr(&e).as_str(), "(let a=(2.0);a)");

    let e = parse_expr("[1, 2, 3]").unwrap();
    assert_eq!(print_expr(&e).as_str(), "[1,2,3]");

    let e = parse_expr("[1.0, 2.0, 3.0]").unwrap();
    assert_eq!(print_expr(&e).as_str(), "[1.0,2.0,3.0]");

    let e = parse_expr("|a, b| a + b").unwrap();
    assert_eq!(print_expr(&e).as_str(), "|a,b|(a+b)");

    let e = parse_expr("for(d, appender, |e| e+1)").unwrap();
    assert_eq!(print_expr(&e).as_str(), "for(d,appender[?],|e|(e+1))");
}

#[test]
fn compare_expressions() {
    let e1 = parse_expr("23").unwrap();
    let e2 = parse_expr("23").unwrap();
    assert!(e1.compare(&e2));

    let e2 = parse_expr("23.0").unwrap();
    assert!(!e1.compare(&e2));

    let e1 = parse_expr("23 + 24").unwrap();
    let e2 = parse_expr("23 + 24").unwrap();
    assert!(e1.compare(&e2));

    let e2 = parse_expr("24 + 23").unwrap();
    assert!(!e1.compare(&e2));

    let e1 = parse_expr("for([1,2], appender, |e| e+1)").unwrap();
    let e2 = parse_expr("for([1,2], appender, |e| e+1)").unwrap();
    assert!(e1.compare(&e2));

    let e2 = parse_expr("for([1,2], appender, |f| f+1)").unwrap();
    //assert!(e1.compare_ignoring_symbols(&e2));

    let e1 = parse_expr("let a = 2; a").unwrap();
    let e2 = parse_expr("let b = 2; b").unwrap();
    assert!(e1.compare_ignoring_symbols(&e2));
        
    let e2 = parse_expr("let b = 2; c").unwrap();
    assert!(!e1.compare_ignoring_symbols(&e2));

    // Undefined symbols cause equality check to return false.
    let e1 = parse_expr("[1, 2, 3, d]").unwrap();
    let e2 = parse_expr("[1, 2, 3, d]").unwrap();
    assert!(!e1.compare_ignoring_symbols(&e2));

    // Symbols can be substituted, so equal.
    let e1 = parse_expr("|a, b| a + b").unwrap();
    let e2 = parse_expr("|c, d| c + d").unwrap();
    assert!(e1.compare_ignoring_symbols(&e2));

    // Symbols don't match up.
    let e2 = parse_expr("|c, d| d + c").unwrap();
    assert!(!e1.compare_ignoring_symbols(&e2));
}

 

#[test]
fn parse_and_print_typed_expressions() {
    let e = parse_expr("a").unwrap();
    assert_eq!(print_typed_expr(&e).as_str(), "a:?");

    let e = Expr {
        kind: ExprKind::Ident(Symbol{name: "a".to_string(), id: 1}),
        ty: Unknown
    };
    assert_eq!(print_typed_expr(&e).as_str(), "a#1:?");

    let e = parse_expr("a:i32").unwrap();
    assert_eq!(print_typed_expr(&e).as_str(), "a:i32");

    let mut e = parse_expr("let a = 2; a").unwrap();
    assert_eq!(print_typed_expr(&e).as_str(), "(let a:?=(2);a:?)");
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr(&e).as_str(), "(let a:i32=(2);a:i32)");

    let mut e = parse_expr("let a = 2; let a = false; a").unwrap();
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr(&e).as_str(), "(let a:i32=(2);(let a:bool=(false);a:bool))");

    // Types should propagate from function parameters to body
    let mut e = parse_expr("|a:i32, b:i32| a + b").unwrap();
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr(&e).as_str(), "|a:i32,b:i32|(a:i32+b:i32)");

    let mut e = parse_expr("|a:f32, b:f32| a + b").unwrap();
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr(&e).as_str(), "|a:f32,b:f32|(a:f32+b:f32)");

    let mut e = parse_expr("let a = [1, 2, 3]; 1").unwrap();
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr(&e).as_str(), "(let a:vec[i32]=([1,2,3]);1)");

    // Mismatched types in MakeVector
    let mut e = parse_expr("[1, true]").unwrap();
    assert!(infer_types(&mut e).is_err());

    let mut e = parse_expr("for([1],appender[?],|b,x|merge(b,x))").unwrap();
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr(&e).as_str(),
        "for([1],appender[i32],|b:appender[i32],x:i32|merge(b:appender[i32],x:i32))");
}



#[test]
fn simple_vertical_loop_fusion() {
    // Two loops.
    let mut e1 = typed_expression("for(result(for([1,2,3], appender, |b,e| merge(b,e+2))), appender, |b,f| merge(b, f+1))");
    fuse_loops(&mut e1).unwrap();
    let e2 = typed_expression("for([1,2,3], appender, |b,e| merge(b, (e+2)+1))");
    assert!(e1.compare_ignoring_symbols(&e2));

    // Three loops.
    let mut e1 = typed_expression("for(result(for(result(for([1,2,3], appender, |b,e| merge(b,e+3))), appender, |b,e| merge(b,e+2))), appender, |b,f| merge(b, f+1))");
    fuse_loops(&mut e1).unwrap();
    let e2 = typed_expression("for([1,2,3], appender, |b,e| merge(b, (((e+3)+2)+1)))");
    assert!(e1.compare_ignoring_symbols(&e2));

    // Make sure correct builder is chosen.
    let mut e1 = typed_expression("for(result(for([1,2,3], appender[i32], |b,e| merge(b,e+2))), appender[f64], |b,f| merge(b, 1.0))");
    fuse_loops(&mut e1).unwrap();
    let e2 = typed_expression("for([1,2,3], appender[f64], |b,e| merge(b, 1.0))");
    assert!(e1.compare_ignoring_symbols(&e2));
}
