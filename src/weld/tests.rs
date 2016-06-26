use super::ast::*;
use super::ast::Expr::*;
use super::ast::Type::*;
use super::ast::ScalarKind::*;
use super::ast::BinOpKind::*;
use super::eval::*;
use super::grammar::parse_expr;
use super::type_inference::*;
use super::pretty_print::*;

#[test]
fn basic_evaluate() {
    fn check(expr: &Expr, expected: i32) {
        let res = evaluate(expr).unwrap().downcast::<i32>().unwrap();
        assert_eq!(*res, expected);
    }

    let e0 = Box::new(I32Literal(0));
    let e1 = Box::new(I32Literal(1));
    let e2 = Box::new(I32Literal(2));
    let e3 = Box::new(I32Literal(3));
    let sym1 = Symbol("sym1".to_string());

    check(&e0, 0);

    let add = BinOp(Scalar(I32), Add, e1.clone(), e2.clone());
    check(&add, 3);

    let sub = BinOp(Scalar(I32), Subtract, e1.clone(), e2.clone());
    check(&sub, -1);

    let mul = BinOp(Scalar(I32), Multiply, e1.clone(), e2.clone());
    check(&mul, 2);

    let div = BinOp(Scalar(I32), Divide, e2.clone(), e1.clone());
    check(&div, 2);

    let div0 = BinOp(Scalar(I32), Divide, e2.clone(), e0.clone());
    assert!(evaluate(&div0).is_err());

    let id1 = Box::new(Ident(Scalar(I32), sym1.clone()));
    let body1 = Box::new(BinOp(Scalar(I32), Add, id1.clone(), id1.clone())); 
    let let1 = Box::new(Let {
        out_type: Scalar(I32),
        symbol: sym1.clone(),
        value: e1.clone(),
        body: body1.clone()
    });
    check(&let1, 2);
    assert!(evaluate(&body1).is_err());

    let body2 = Box::new(BinOp(Scalar(I32), Add, let1.clone(), id1.clone()));
    let let2 = Box::new(Let {
        out_type: Scalar(I32),
        symbol: sym1.clone(),
        value: e3.clone(),
        body: body2.clone()
    });
    check(&let2, 5);
    assert!(evaluate(&body2).is_err());
}

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

    let e = parse_expr("map(d, |e| e+1)").unwrap();
    assert_eq!(print_expr(e.as_ref()).as_str(), "map(d,|e|(e+1))");
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

    let mut e = parse_expr("let d=map([1],|x|x+1);1").unwrap();
    infer_types(&mut e).unwrap();
    assert_eq!(print_typed_expr(e.as_ref()).as_str(),
        "let d:vec[i32]=(map([1],|x:i32|(x:i32+1)));1");
}