#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Literal(i32),
    BinOp(BinOpType, Box<Expr>, Box<Expr>)
}

#[derive(Clone, Debug, PartialEq)]
pub enum BinOpType {
    Add, Subtract, Multiply, Divide
}