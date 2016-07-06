use weld_ast::Type::*;
use weld_ast::ScalarKind::*;
use super::*;

#[test]
fn types() {
    let mut ctx = GeneratorContext;
    assert_eq!(ctx.llvm_type(&Scalar(I32)).unwrap(), "i32");
    assert_eq!(ctx.llvm_type(&Scalar(Bool)).unwrap(), "i1");

    //let weld_type = parse_type("{i32,bool,i32}").unwrap().to_type().unwrap();
    //assert_eq!(ctx.llvm_type(&weld_type).unwrap(), "%s1");
}