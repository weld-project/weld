use weld_ast::pretty_print::*;
use weld_transform::type_inference::*;
use weld_parser::parse_expr;

#[test]
fn parse_and_print_literal_expressions() {
    // Test i32 literal expressions.
    let e = parse_expr("23").unwrap();
    assert_eq!(print_expr(&e).as_str(), "23");

    let e = parse_expr("0b111").unwrap();
    assert_eq!(print_expr(&e).as_str(), "7");

    let e = parse_expr("0xff").unwrap();
    assert_eq!(print_expr(&e).as_str(), "255");

    let e = parse_expr("0o10").unwrap();
    assert_eq!(print_expr(&e).as_str(), "8");

    // Test i64 literal expressions.
    let e = parse_expr("23L").unwrap();
    assert_eq!(print_expr(&e).as_str(), "23L");

    let e = parse_expr("0b111L").unwrap();
    assert_eq!(print_expr(&e).as_str(), "7L");

    let e = parse_expr("0xffL").unwrap();
    assert_eq!(print_expr(&e).as_str(), "255L");

    let e = parse_expr("0o10L").unwrap();
    assert_eq!(print_expr(&e).as_str(), "8L");

    // Test f32 literal expressions.
    let e = parse_expr("23.0").unwrap();
    assert_eq!(print_expr(&e).as_str(), "23.0");

    let e = parse_expr("23.5").unwrap();
    assert_eq!(print_expr(&e).as_str(), "23.5");

    let e = parse_expr("23e5").unwrap();
    assert_eq!(print_expr(&e).as_str(), "2300000.0");

    let e = parse_expr("23.5e5").unwrap();
    assert_eq!(print_expr(&e).as_str(), "2350000.0");

    // Test f64 literal expressions.
    let e = parse_expr("23.0f").unwrap();
    assert_eq!(print_expr(&e).as_str(), "23.0F");

    let e = parse_expr("23.5f").unwrap();
    assert_eq!(print_expr(&e).as_str(), "23.5F");

    let e = parse_expr("23e5f").unwrap();
    assert_eq!(print_expr(&e).as_str(), "2300000.0F");

    let e = parse_expr("23.5e5f").unwrap();
    assert_eq!(print_expr(&e).as_str(), "2350000.0F");

    // Test boolean.
    let e = parse_expr("true").unwrap();
    assert_eq!(print_expr(&e).as_str(), "true");

    // Test overflow of 32-bit variants.
    assert!(parse_expr("999999999999999").is_err());  // i32 literal too big
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
    assert_eq!(print_expr(&e).as_str(), "let a=(2);a");

    let e = parse_expr("let a = 2.0; a").unwrap();
    assert_eq!(print_expr(&e).as_str(), "let a=(2.0);a");

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
fn parse_and_print_typed_expressions() {
    let e = parse_expr("a").unwrap();
    assert_eq!(print_typed_expr(&e).as_str(), "a:?");

    let e = parse_expr("a:i32").unwrap();
    assert_eq!(print_typed_expr(&e).as_str(), "a:i32");

    let mut e = parse_expr("let a = 2; a").unwrap();
    assert_eq!(print_typed_expr(&e).as_str(), "let a:?=(2);a:?");
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr(&e).as_str(), "let a:i32=(2);a:i32");

    let mut e = parse_expr("let a = 2; let a = false; a").unwrap();
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr(&e).as_str(), "let a:i32=(2);let a:bool=(false);a:bool");

    // Types should propagate from function parameters to body 
    let mut e = parse_expr("|a:i32, b:i32| a + b").unwrap();
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr(&e).as_str(), "|a:i32,b:i32|(a:i32+b:i32)");

    let mut e = parse_expr("|a:f32, b:f32| a + b").unwrap();
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr(&e).as_str(), "|a:f32,b:f32|(a:f32+b:f32)");

    let mut e = parse_expr("let a = [1, 2, 3]; 1").unwrap();
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr(&e).as_str(), "let a:vec[i32]=([1,2,3]);1");

    // Mismatched types in MakeVector 
    let mut e = parse_expr("[1, true]").unwrap();
    assert!(infer_types(&mut e).is_err());

    let mut e = parse_expr("for([1],appender,|b,x|merge(b,x))").unwrap();
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr(&e).as_str(),
        "for([1],appender[i32],|b:appender[i32],x:i32|merge(b:appender[i32],x:i32))");
}
