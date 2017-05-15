use super::ast::{Expr, Type, ExprKind, Symbol};
use super::partial_types::PartialType::Unknown;
use super::parser::parse_expr;
use super::pretty_print::*;
use super::type_inference::*;

// Transforms. TODO(shoumik) move these tests somewhere else?
use super::transforms::fuse_loops_vertical;
use super::transforms::fuse_loops_horizontal;
use super::transforms::inline_let;
use super::transforms::uniquify;

/// Returns a typed expression.
#[cfg(test)]
fn typed_expression(s: &str) -> Expr<Type> {
    let mut e1 = parse_expr(s).unwrap();
    infer_types(&mut e1).unwrap();
    e1.to_typed().unwrap()
}

#[test]
fn parse_and_print_literal_expressions() {
    let tests = vec![// i32 literal expressions
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
                     ("false", "false")];

    for test in tests {
        let e = parse_expr(test.0).unwrap();
        assert_eq!(print_expr_without_indent(&e).as_str(), test.1);
    }

    // Test overflow of integer types
    assert!(parse_expr("999999999999999").is_err()); // i32 literal too big
    assert!(parse_expr("999999999999999L").is_ok());
    assert!(parse_expr("999999999999999999999999999999L").is_err()); // i64 literal too big
}

#[test]
fn parse_and_print_simple_expressions() {
    let e = parse_expr("23 + 32").unwrap();
    assert_eq!(print_expr_without_indent(&e).as_str(), "(23+32)");

    let e = parse_expr("2 - 3 - 4").unwrap();
    assert_eq!(print_expr_without_indent(&e).as_str(), "((2-3)-4)");

    let e = parse_expr("2 - (3 - 4)").unwrap();
    assert_eq!(print_expr_without_indent(&e).as_str(), "(2-(3-4))");

    let e = parse_expr("a").unwrap();
    assert_eq!(print_expr_without_indent(&e).as_str(), "a");

    let e = parse_expr("let a = 2; a").unwrap();
    assert_eq!(print_expr_without_indent(&e).as_str(), "(let a=(2);a)");

    let e = parse_expr("let a = 2.0; a").unwrap();
    assert_eq!(print_expr_without_indent(&e).as_str(), "(let a=(2.0);a)");

    let e = parse_expr("[1, 2, 3]").unwrap();
    assert_eq!(print_expr_without_indent(&e).as_str(), "[1,2,3]");

    let e = parse_expr("[1.0, 2.0, 3.0]").unwrap();
    assert_eq!(print_expr_without_indent(&e).as_str(), "[1.0,2.0,3.0]");

    let e = parse_expr("|a, b| a + b").unwrap();
    assert_eq!(print_expr_without_indent(&e).as_str(), "|a,b|(a+b)");

    let e = parse_expr("for(d, appender, |e| e+1)").unwrap();
    assert_eq!(print_expr_without_indent(&e).as_str(),
               "for(d,appender[?],|e|(e+1))");
}

#[test]
fn parse_and_print_uniquified_expressions() {
    let mut e = parse_expr("let a = 2; a").unwrap();
    let _ = uniquify(&mut e);
    assert_eq!(print_expr_without_indent(&e).as_str(), "(let a=(2);a)");

    // Redefine a symbol.
    let mut e = parse_expr("let a = 2; let a = 3; a").unwrap();
    let _ = uniquify(&mut e);
    assert_eq!(print_expr_without_indent(&e).as_str(),
               "(let a=(2);(let a#1=(3);a#1))");

    // Make sure Let values aren't renamed.
    let mut e = parse_expr("let a = 2; let a = a+1; a").unwrap();
    let _ = uniquify(&mut e);
    assert_eq!(print_expr_without_indent(&e).as_str(),
               "(let a=(2);(let a#1=((a+1));a#1))");

    // Lambdas and proper scoping.
    let mut e = parse_expr("let a = 2; (|a,b|a+b)(1,2) + a").unwrap();
    let _ = uniquify(&mut e);
    assert_eq!(print_expr_without_indent(&e).as_str(),
               "(let a=(2);((|a#1,b|(a#1+b))(1,2)+a))");
}

#[test]
fn parse_and_print_for_expressions() {
    let e = parse_expr("for(d, appender, |e| e+1)").unwrap();
    assert_eq!(print_expr_without_indent(&e).as_str(),
               "for(d,appender[?],|e|(e+1))");

    let e = parse_expr("for(iter(d), appender, |e| e+1)").unwrap();
    assert_eq!(print_expr_without_indent(&e).as_str(),
               "for(d,appender[?],|e|(e+1))");

    let e = parse_expr("for(iter(d,0L,4L,1L), appender, |e| e+1)").unwrap();
    assert_eq!(print_expr_without_indent(&e).as_str(),
               "for(iter(d,0L,4L,1L),appender[?],|e|(e+1))");

    let e = parse_expr("for(zip(d), appender, |e| e+1)").unwrap();
    assert_eq!(print_expr_without_indent(&e).as_str(),
               "for(d,appender[?],|e|(e+1))");

    let e = parse_expr("for(zip(d,e), appender, |e| e+1)").unwrap();
    assert_eq!(print_expr_without_indent(&e).as_str(),
               "for(zip(d,e),appender[?],|e|(e+1))");

    let e = parse_expr("for(zip(a,b,iter(c,0L,4L,1L),iter(d)), appender, |e| e+1)").unwrap();
    assert_eq!(print_expr_without_indent(&e).as_str(),
               "for(zip(a,b,iter(c,0L,4L,1L),d),appender[?],|e|(e+1))");
}

#[test]
fn parse_and_print_typed_expr_without_indentessions() {
    let e = parse_expr("a").unwrap();
    assert_eq!(print_typed_expr_without_indent(&e).as_str(), "a:?");

    let e = Expr {
        kind: ExprKind::Ident(Symbol {
            name: "a".to_string(),
            id: 1,
        }),
        ty: Unknown,
    };
    assert_eq!(print_typed_expr_without_indent(&e).as_str(), "a#1:?");

    let e = parse_expr("a:i32").unwrap();
    assert_eq!(print_typed_expr_without_indent(&e).as_str(), "a:i32");

    let mut e = parse_expr("let a = 2; a").unwrap();
    assert_eq!(print_typed_expr_without_indent(&e).as_str(),
               "(let a:?=(2);a:?)");
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr_without_indent(&e).as_str(),
               "(let a:i32=(2);a:i32)");

    let mut e = parse_expr("let a = 2; let a = false; a").unwrap();
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr_without_indent(&e).as_str(),
               "(let a:i32=(2);(let a:bool=(false);a:bool))");

    // Types should propagate from function parameters to body
    let mut e = parse_expr("|a:i32, b:i32| a + b").unwrap();
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr_without_indent(&e).as_str(),
               "|a:i32,b:i32|(a:i32+b:i32)");

    let mut e = parse_expr("|a:f32, b:f32| a + b").unwrap();
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr_without_indent(&e).as_str(),
               "|a:f32,b:f32|(a:f32+b:f32)");

    let mut e = parse_expr("let a = [1, 2, 3]; 1").unwrap();
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr_without_indent(&e).as_str(),
               "(let a:vec[i32]=([1,2,3]);1)");

    // Mismatched types in MakeVector
    let mut e = parse_expr("[1, true]").unwrap();
    assert!(infer_types(&mut e).is_err());

    let mut e = parse_expr("for([1],appender[?],|b,i,x|merge(b,x))").unwrap();
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr_without_indent(&e).as_str(),
               "for([1],appender[i32],|b:appender[i32],i:i64,x:i32|merge(b:appender[i32],x:i32))");
}

#[test]
fn compare_expressions() {
    let e1 = parse_expr("for([1,2], appender, |e| e+1)").unwrap();
    let e2 = parse_expr("for([1,2], appender, |f| f+1)").unwrap();
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    let e1 = parse_expr("let a = 2; a").unwrap();
    let e2 = parse_expr("let b = 2; b").unwrap();
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    let e2 = parse_expr("let b = 2; c").unwrap();
    assert!(!e1.compare_ignoring_symbols(&e2).unwrap());

    // Undefined symbols cause equality check to return false.
    let e1 = parse_expr("[1, 2, 3, d]").unwrap();
    let e2 = parse_expr("[1, 2, 3, d]").unwrap();
    assert!(e1.compare_ignoring_symbols(&e2).is_err());

    // Symbols can be substituted, so equal.
    let e1 = parse_expr("|a, b| a + b").unwrap();
    let e2 = parse_expr("|c, d| c + d").unwrap();
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Symbols don't match up.
    let e2 = parse_expr("|c, d| d + c").unwrap();
    assert!(!e1.compare_ignoring_symbols(&e2).unwrap());
}

#[test]
fn simple_horizontal_loop_fusion() {
    // Two loops.
    let mut e1 = typed_expression("for(zip(
            result(for([1,2,3], appender, |b,i,e| merge(b, e+1))),
            result(for([1,2,3], appender,|b2,i2,e2| merge(b2,e2+1)))
        ), appender, |b,i,e| merge(b, e.$0+1))");
    fuse_loops_horizontal(&mut e1);
    let e2 = typed_expression("for(result(for([1,2,3], appender, |b,i,e| merge(b, {e+1,e+1}))), \
                               appender, |b,i,e| merge(b, e.$0+1))");
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Three loops.
    let mut e1 = typed_expression("for(zip(
            result(for([1,2,3], appender, |b,i,e| merge(b, e+1))),
            result(for([1,2,3], appender,|b2,i2,e2| merge(b2,e2+2))),
            result(for([1,2,3], appender,|b3,i3,e3| merge(b3,e3+3)))
        ), appender, |b,i,e| merge(b, e.$0+1))");
    fuse_loops_horizontal(&mut e1);
    let e2 = typed_expression("for(result(for([1,2,3], appender, |b,i,e| merge(b, \
                               {e+1,e+2,e+3}))), appender, |b,i,e| merge(b, e.$0+1))");
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Iters in inner loop
    let mut e1 = typed_expression("for(zip(
            result(for(iter([1,2,3], 0L, 2L, 1L), appender, |b,i,e| merge(b, e+1))),
            result(for(iter([1,2,3], 0L, 2L, 1L), appender, |b,i,e| merge(b, e+2)))
        ), appender, |b,i,e| merge(b, e.$0+1))");
    fuse_loops_horizontal(&mut e1);
    let e2 = typed_expression("for(result(for(iter([1,2,3], 0L, 2L, 1L), appender, |b,i,e| \
                               merge(b, {e+1,e+2}))), appender, |b,i,e| merge(b, e.$0+1))");
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Iters in outer loop.
    let mut e1 = typed_expression("for(zip(
            iter(result(for([1,2,3], appender, |b,i,e| merge(b, e+1))), 0L, 2L, 1L),
            iter(result(for([1,2,3], appender, |b,i,e| merge(b, e+2))), 0L, 2L, 1L)
        ), appender, |b,i,e| merge(b, e.$0+1))");
    fuse_loops_horizontal(&mut e1);
    let e2 = typed_expression("for(iter(result(for([1,2,3], appender, |b,i,e| merge(b, \
                               {e+1,e+2}))), 0L, 2L, 1L), appender, |b,i,e| merge(b, e.$0+1))");
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Two loops with different vectors; should fail.
    let mut e1 = typed_expression("for(zip(
            result(for([1,2,3], appender, |b,i,e| merge(b, e+1))),
            result(for([1,2,4], appender,|b2,i2,e2| merge(b2,e2+1)))
        ), appender, |b,i,e| merge(b, e.$0+1))");
    let e2 = e1.clone();
    fuse_loops_horizontal(&mut e1);
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());
}

#[test]
fn simple_vertical_loop_fusion() {
    // Two loops.
    let mut e1 = typed_expression("for(result(for([1,2,3], appender, |b,i,e| merge(b,e+2))), \
                                   appender, |b,h,f| merge(b, f+1))");
    fuse_loops_vertical(&mut e1);
    let e2 = typed_expression("for([1,2,3], appender, |b,i,e| merge(b, (e+2)+1))");
    println!("{}", print_expr_without_indent(&e1));
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Three loops.
    let mut e1 = typed_expression("for(result(for(result(for([1,2,3], appender, |b,i,e| \
                                   merge(b,e+3))), appender, |b,i,e| merge(b,e+2))), appender, \
                                   |b,h,f| merge(b, f+1))");
    fuse_loops_vertical(&mut e1);
    let e2 = typed_expression("for([1,2,3], appender, |b,i,e| merge(b, (((e+3)+2)+1)))");
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Merges in other positions, replace builder identifiers.
    let mut e1 = typed_expression("for(result(for([1,2,3], appender, |b,i,e| if(e>5, \
                                   merge(b,e+2), b))), appender, |b,h,f| merge(b, f+1))");
    fuse_loops_vertical(&mut e1);
    let e2 = typed_expression("for([1,2,3], appender, |b,i,e| if(e>5, merge(b, (e+2)+1), b))");
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Make sure correct builder is chosen.
    let mut e1 = typed_expression("for(result(for([1,2,3], appender[i32], |b,i,e| \
                                   merge(b,e+2))), appender[f64], |b,h,f| merge(b, 1.0))");
    fuse_loops_vertical(&mut e1);
    let e2 = typed_expression("for([1,2,3], appender[f64], |b,i,e| merge(b, 1.0))");
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Multiple inner loops.
    let mut e1 = typed_expression("for(result(for(zip([1,2,3],[4,5,6]), appender, |b,i,e| \
                                   merge(b,e.$0+2))), appender, |b,h,f| merge(b, f+1))");
    fuse_loops_vertical(&mut e1);
    let e2 = typed_expression("for(zip([1,2,3],[4,5,6]), appender, |b,i,e| merge(b, (e.$0+2)+1))");
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Iter where inner data consumed fully.
    let mut e1 = typed_expression("let a = [1,2,3]; for(result(for(iter(a, 0L, len(a), 1L), \
                                   appender, |b,i,e| merge(b,e+2))), appender, |b,h,f| merge(b, \
                                   f+1))");
    fuse_loops_vertical(&mut e1);
    let e2 = typed_expression("let a = [1,2,3]; for(iter(a,0L,len(a),1L), appender, |b,i,e| \
                               merge(b, (e+2)+1))");
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Inner data not consumed fully.
    let mut e1 = typed_expression("for(result(for(iter([1,2,3], 0L, 1L, 1L), appender, |b,i,e| \
                                   merge(b,e+2))), appender, |b,h,f| merge(b, f+1))");
    fuse_loops_vertical(&mut e1);
    // Loop fusion should fail.
    let e2 = typed_expression("for(result(for(iter([1,2,3], 0L, 1L, 1L), appender, |b,i,e| \
                               merge(b,e+2))), appender, |b,h,f| merge(b, f+1))");
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());
}

#[test]
fn inline_lets() {
    let mut e1 = typed_expression("let a = 1; a + 2");
    inline_let(&mut e1);
    let e2 = typed_expression("1 + 2");
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    let mut e1 = typed_expression("let a = 1; a + a + 2");
    // The transform should fail since the identifier is used more than once.
    inline_let(&mut e1);
    let e2 = typed_expression("let a = 1; a + a + 2");
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    let mut e1 = typed_expression("let a = 1L; for([1L,2L,3L], appender, |b,i,e| merge(b, e + a \
                                   + 2L))");
    inline_let(&mut e1);
    // The transform should fail since the identifier is used in a loop.
    let e2 = typed_expression("let a = 1L; for([1L,2L,3L], appender, |b,i,e| merge(b, e + a + \
                               2L))");
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    let mut e1 = typed_expression("let a = 1; let b = 2; let c = 3; a + b + c");
    inline_let(&mut e1);
    let e2 = typed_expression("1 + 2 + 3");
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());
}
