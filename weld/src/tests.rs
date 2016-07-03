use weld_parser::parse_expr;
use weld_ast::pretty_print::*;
use super::type_inference::*;

#[test]
fn parse_and_print_expressions() {
    let e = parse_expr("23").unwrap();
    assert_eq!(print_expr(e.as_ref()).as_str(), "23");

    let e = parse_expr("true").unwrap();
    assert_eq!(print_expr(e.as_ref()).as_str(), "true");

    assert!(parse_expr("999999999999999").is_err());  // i32 literal too big

    let e = parse_expr("23 + 32").unwrap();
    assert_eq!(print_expr(e.as_ref()).as_str(), "(23+32)");

    let e = parse_expr("2 - 3 - 4").unwrap();
    assert_eq!(print_expr(e.as_ref()).as_str(), "((2-3)-4)");

    let e = parse_expr("2 - (3 - 4)").unwrap();
    assert_eq!(print_expr(e.as_ref()).as_str(), "(2-(3-4))");

    let e = parse_expr("a").unwrap();
    assert_eq!(print_expr(e.as_ref()).as_str(), "a");

    let e = parse_expr("let a = 2; a").unwrap();
    assert_eq!(print_expr(e.as_ref()).as_str(), "let a=(2);a");

    let e = parse_expr("[1, 2, 3]").unwrap();
    assert_eq!(print_expr(e.as_ref()).as_str(), "[1,2,3]");

    let e = parse_expr("|a, b| a + b").unwrap();
    assert_eq!(print_expr(e.as_ref()).as_str(), "|a,b|(a+b)");

    let e = parse_expr("for(d, appender, |e| e+1)").unwrap();
    assert_eq!(print_expr(e.as_ref()).as_str(), "for(d,appender[?],|e|(e+1))");
}

#[test]
fn parse_and_print_typed_expressions() {
    let e = *parse_expr("a").unwrap();
    assert_eq!(print_typed_expr(&e).as_str(), "a:?");

    let e = *parse_expr("a:i32").unwrap();
    assert_eq!(print_typed_expr(&e).as_str(), "a:i32");

    let mut e = parse_expr("let a = 2; a").unwrap();
    assert_eq!(print_typed_expr(e.as_ref()).as_str(), "let a:?=(2);a:?");
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr(e.as_ref()).as_str(), "let a:i32=(2);a:i32");

    let mut e = parse_expr("let a = 2; let a = false; a").unwrap();
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr(e.as_ref()).as_str(), "let a:i32=(2);let a:bool=(false);a:bool");

    // Types should propagate from function parameters to body 
    let mut e = parse_expr("|a:i32, b:i32| a + b").unwrap();
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr(e.as_ref()).as_str(), "|a:i32,b:i32|(a:i32+b:i32)");

    let mut e = parse_expr("let a = [1, 2, 3]; 1").unwrap();
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr(e.as_ref()).as_str(), "let a:vec[i32]=([1,2,3]);1");

    // Mismatched types in MakeVector 
    let mut e = parse_expr("[1, true]").unwrap();
    assert!(infer_types(&mut e).is_err());

    let mut e = parse_expr("for([1],appender,|b,x|merge(b,x))").unwrap();
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr(e.as_ref()).as_str(),
        "for([1],appender[i32],|b:appender[i32],x:i32|merge(b:appender[i32],x:i32))");
}
