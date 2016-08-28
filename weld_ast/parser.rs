use std::vec::Vec;

use weld_error::*;
use super::ast::Symbol;
use super::ast::BinOpKind::*;
use super::ast::ExprKind::*;
use super::ast::ScalarKind::*;
use super::partial_types::*;
use super::partial_types::PartialBuilderKind::*;
use super::partial_types::PartialType::*;
use super::tokenizer::*;
use super::tokenizer::Token::*;

#[cfg(test)] use super::pretty_print::*;

pub fn parse_expr(input: &str) -> WeldResult<PartialExpr> {
    let tokens = try!(tokenize(input));
    Parser::new(&tokens).expr().map(|b| *b)
}

pub fn parse_type(input: &str) -> WeldResult<PartialType> {
    let tokens = try!(tokenize(input));
    Parser::new(&tokens).partial_type()
}

/// A stateful object that parses a sequence of tokens, tracking its position at each point.
/// Assumes that the tokens end with a TEndOfInput.
struct Parser<'t> {
    tokens: &'t [Token],
    position: usize
}

impl<'t> Parser<'t> {
    fn new(tokens: &[Token]) -> Parser {
        Parser { tokens: tokens, position: 0 }
    }

    /// Look at the next token to be parsed.
    fn peek(&self) -> &'t Token {
        &self.tokens[self.position]
    }

    /// Consume and return the next token. 
    fn next(&mut self) -> &'t Token {
        let token = &self.tokens[self.position];
        self.position += 1;
        token
    }

    /// Parse an expression starting at self.position.
    fn expr(&mut self) -> WeldResult<Box<PartialExpr>> {
        if *self.peek() == TLet {
            self.let_expr()
        } else {
            self.sum_expr()
        }
    }

    fn let_expr(&mut self) -> WeldResult<Box<PartialExpr>> {
        if *self.next() != TLet {
            return weld_err!("Expected 'let'");
        }
        let name = try!(self.name());
        let ty = try!(self.optional_type());
        if *self.next() != TEqual {
            return weld_err!("Expected '='");
        }
        let value = try!(self.sum_expr());
        if *self.next() != TSemicolon {
            return weld_err!("Expected ';'");
        }
        let body = try!(self.expr());
        let mut expr = expr_box(Let(name, value, body));
        expr.ty = ty;
        Ok(expr)
    }

    fn sum_expr(&mut self) -> WeldResult<Box<PartialExpr>> {
        let mut res = try!(self.prod_expr());
        while *self.peek() == TPlus || *self.peek() == TMinus {
            let token = self.next();
            let right = try!(self.prod_expr());
            if *token == TPlus {
                res = expr_box(BinOp(Add, res, right))
            } else {
                res = expr_box(BinOp(Subtract, res, right))
            }
        }
        Ok(res)
    }

    fn prod_expr(&mut self) -> WeldResult<Box<PartialExpr>> {
        let mut res = try!(self.leaf_expr());
        while *self.peek() == TTimes || *self.peek() == TDivide {
            let token = self.next();
            let right = try!(self.leaf_expr());
            if *token == TTimes {
                res = expr_box(BinOp(Multiply, res, right))
            } else {
                res = expr_box(BinOp(Divide, res, right))
            }
        }
        Ok(res)
    }

    fn leaf_expr(&mut self) -> WeldResult<Box<PartialExpr>> {
        match *self.next() {
            TI32Literal(value) => Ok(expr_box(I32Literal(value))),
            TBoolLiteral(value) => Ok(expr_box(BoolLiteral(value))),
            TIdent(ref name) => Ok(expr_box(Ident(name.clone()))),

            TOpenParen => {
                let expr = try!(self.expr());
                if *self.next() != TCloseParen {
                    return weld_err!("Expected ')'")
                }
                Ok(expr)
            },

            TOpenBracket => {
                let mut exprs: Vec<PartialExpr> = Vec::new();
                while *self.peek() != TCloseBracket {
                    let expr = try!(self.expr());
                    exprs.push(*expr);
                    if *self.peek() == TComma {
                        self.next();
                    } else if *self.peek() != TCloseBracket {
                        return weld_err!("Expected ',' or ']'")
                    }
                }
                self.next();  // Will be a TCloseBracket
                Ok(expr_box(MakeVector(exprs)))
            }

            TOpenBrace => {
                let mut exprs: Vec<PartialExpr> = Vec::new();
                while *self.peek() != TCloseBrace {
                    let expr = try!(self.expr());
                    exprs.push(*expr);
                    if *self.peek() == TComma {
                        self.next();
                    } else if *self.peek() != TCloseBrace {
                        return weld_err!("Expected ',' or '}}'")
                    }
                }
                self.next();  // Will be a TCloseBrace
                Ok(expr_box(MakeStruct(exprs)))
            }

            ref other => weld_err!("Expected expression but got '{}'", other)
        }
    }

    fn name(&mut self) -> WeldResult<Symbol> {
        match *self.next() {
            TIdent(ref name) => Ok(name.clone()),
            ref other => weld_err!("Expected identifier but got '{}'", other)
        }
    }

    fn optional_type(&mut self) -> WeldResult<PartialType> {
        if *self.peek() == TColon {
            self.next();   // Skip TColon
            self.partial_type() 
        } else {
            Ok(Unknown)
        }
    }

    fn partial_type(&mut self) -> WeldResult<PartialType> {
        match *self.next() {
            TQuestion => Ok(Unknown),

            TIdent(ref name) => {
                match name.as_ref() {
                    "i32" => Ok(Scalar(I32)),
                    "i64" => Ok(Scalar(I64)),
                    "f32" => Ok(Scalar(F32)),
                    "f64" => Ok(Scalar(F64)),
                    "bool" => Ok(Scalar(Bool)),

                    "vec" => {
                        if *self.next() != TOpenBracket {
                            return weld_err!("Expected '['");
                        }
                        let elem_type = try!(self.partial_type());
                        if *self.next() != TCloseBracket {
                            return weld_err!("Expected ']'");
                        }
                        Ok(Vector(Box::new(elem_type)))
                    }

                    "appender" => {
                        if *self.next() == TOpenBracket {
                            return weld_err!("Expected '['");
                        }
                        let elem_type = try!(self.partial_type());
                        if *self.next() != TCloseBracket {
                            return weld_err!("Expected ']'");
                        }
                        Ok(Builder(Appender(Box::new(elem_type))))
                    }

                    other => weld_err!("Expected type but got '{}'", other)
                }
            },

            TOpenBrace => {
                let mut types: Vec<PartialType> = Vec::new();
                while *self.peek() != TCloseBrace {
                    let ty = try!(self.partial_type());
                    types.push(ty);
                    if *self.peek() == TComma {
                        self.next();
                    } else if *self.peek() != TCloseBrace {
                        return weld_err!("Expected ',' or '}}'")
                    }
                }
                self.next();  // Will be a TCloseBrace
                Ok(Struct(types))
            }

            ref other => weld_err!("Expected type but got '{}'", other)
        }
    }
}

#[test]
fn basic_parsing() {
    let e = parse_expr("10 - 2 - 3 + 1").unwrap();
    assert_eq!(print_expr(&e), "(((10-2)-3)+1)");

    let e = parse_expr("10 * 2 - 4 - 3 / 1").unwrap();
    assert_eq!(print_expr(&e), "(((10*2)-4)-(3/1))");

    let e = parse_expr("[1, 2+3, 2]").unwrap();
    assert_eq!(print_expr(&e), "[1,(2+3),2]");

    let e = parse_expr("let a = 3+2; let b = (let c=a; c); b").unwrap();
    assert_eq!(print_expr(&e), "let a=((3+2));let b=(let c=(a);c);b");

    let e = parse_expr("let a: vec[i32] = [2, 3]; a").unwrap();
    assert_eq!(print_expr(&e), "let a=([2,3]);a");

    assert!(parse_expr("10 * * 2").is_err());

    let t = parse_type("{i32, vec[vec[?]], ?}").unwrap();
    assert_eq!(print_type(&t), "{i32,vec[vec[?]],?}");
}