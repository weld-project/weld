extern crate easy_ll;

use super::*;
use super::ast::*;

#[test]
fn basics() {
    let a = Expr::Literal(5);
    assert_eq!(a, Expr::Literal(5));
    easy_ll::hello_llvm();
}
