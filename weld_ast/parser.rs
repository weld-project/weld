use std::vec::Vec;

use weld_error::*;
use super::ast::BinOpKind::*;
use super::ast::ExprKind::*;
use super::partial_types::*;
use super::tokenizer::*;
use super::tokenizer::Token::*;

#[cfg(test)] use super::pretty_print::*;

pub fn parse_expr(input: &str) -> WeldResult<PartialExpr> {
    let tokens = try!(tokenize(input));
    Parser::new(&tokens).sum_expr().map(|b| *b)
}

struct Parser<'t> {
    tokens: &'t [Token],
    position: usize
}

impl<'t> Parser<'t> {
    fn new(tokens: &[Token]) -> Parser {
        Parser { tokens: tokens, position: 0 }
    }

    fn peek(&self) -> &'t Token {
        &self.tokens[self.position]
    }

    fn next(&mut self) -> &'t Token {
        let token = &self.tokens[self.position];
        self.position += 1;
        token
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
        let mut res = try!(self.terminal());
        while *self.peek() == TTimes || *self.peek() == TDivide {
            let token = self.next();
            let right = try!(self.terminal());
            if *token == TTimes {
                res = expr_box(BinOp(Multiply, res, right))
            } else {
                res = expr_box(BinOp(Divide, res, right))
            }
        }
        Ok(res)
    }

    fn terminal(&mut self) -> WeldResult<Box<PartialExpr>> {
        match *self.next() {
            TI32Lit(value) => Ok(expr_box(I32Literal(value))),
            TIdent(ref name) => Ok(expr_box(Ident(name.clone()))),

            TOpenParen => {
                let expr = try!(self.sum_expr());
                if *self.next() != TCloseParen {
                    return weld_err!("Expected ')'")
                }
                Ok(expr)
            },

            TOpenBracket => {
                let mut exprs: Vec<PartialExpr> = Vec::new();
                while *self.peek() != TCloseBracket {
                    let expr = try!(self.sum_expr());
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

            ref other => weld_err!("Expected literal but got {:?}", other)
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

    assert!(parse_expr("10 * * 2").is_err());
}