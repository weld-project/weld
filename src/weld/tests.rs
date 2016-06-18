use super::ast::*;
use super::ast::Expr::*;
use super::ast::Type::*;
use super::ast::ScalarKind::*;
use super::ast::BinOpKind::*;
use super::eval::*;

#[test]
fn basic_evaluate() {
    fn check(expr: &Expr, expected: i32) {
        let res = evaluate(expr).unwrap().downcast::<i32>().unwrap();
        assert_eq!(*res, expected);
    }

    let e0 = I32Literal(0);
    let e1 = I32Literal(1);
    let e2 = I32Literal(2);
    let e3 = I32Literal(3);
    let sym1 = Symbol("sym1".to_string());

    check(&e0, 0);

    let add = BinOp(Scalar(I32), Add, Box::new(e1.clone()), Box::new(e2.clone()));
    check(&add, 3);

    let sub = BinOp(Scalar(I32), Subtract, Box::new(e1.clone()), Box::new(e2.clone()));
    check(&sub, -1);

    let mul = BinOp(Scalar(I32), Multiply, Box::new(e1.clone()), Box::new(e2.clone()));
    check(&mul, 2);

    let div = BinOp(Scalar(I32), Divide, Box::new(e2.clone()), Box::new(e1.clone()));
    check(&div, 2);

    let div0 = BinOp(Scalar(I32), Divide, Box::new(e2.clone()), Box::new(e0.clone()));
    assert!(evaluate(&div0).is_err());

    let id1 = Ident(Scalar(I32), sym1.clone());
    let body1 = BinOp(Scalar(I32), Add, Box::new(id1.clone()), Box::new(id1.clone())); 
    let let1 = Let {
        out_type: Scalar(I32),
        symbol: sym1.clone(),
        value: Box::new(e1.clone()),
        body: Box::new(body1.clone())
    };
    check(&let1, 2);
    assert!(evaluate(&body1).is_err());

    let body2 = BinOp(Scalar(I32), Add, Box::new(let1.clone()), Box::new(id1.clone()));
    let let2 = Let {
        out_type: Scalar(I32),
        symbol: sym1.clone(),
        value: Box::new(e3.clone()),
        body: Box::new(body2.clone())
    };
    check(&let2, 5);
    assert!(evaluate(&body2).is_err());
}
