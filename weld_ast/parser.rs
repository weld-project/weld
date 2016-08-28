//! Top-down recursive descent parser for Weld.
//!
//! TODO:
//! - Ascribe expression
//! - Dot
//! - Apply

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

    /// Consume the next token and check that it equals `expected`. If not, return an Err.
    fn consume(&mut self, expected: Token) -> WeldResult<()> {
        if *self.next() != expected {
            weld_err!("Expected '{}'", expected)
        } else {
            Ok(())
        }
    }

    /// Parse an expression starting at the current position.
    fn expr(&mut self) -> WeldResult<Box<PartialExpr>> {
        if *self.peek() == TLet {
            self.let_expr()
        } else if *self.peek() == TBar {
            self.lambda_expr()
        } else {
            self.sum_expr()
        }
    }

    /// Parse 'let name = value; body' starting at the current position.
    fn let_expr(&mut self) -> WeldResult<Box<PartialExpr>> {
        try!(self.consume(TLet));
        let name = try!(self.name());
        let ty = try!(self.optional_type());
        try!(self.consume(TEqual));
        let value = try!(self.sum_expr());
        try!(self.consume(TSemicolon));
        let body = try!(self.expr());
        let mut expr = expr_box(Let(name, value, body));
        expr.ty = ty;
        Ok(expr)
    }

    /// Parse '|params| body' starting at the current position.
    fn lambda_expr(&mut self) -> WeldResult<Box<PartialExpr>> {
        try!(self.consume(TBar));
        let mut params: Vec<PartialParameter> = Vec::new();
        while *self.peek() != TBar {
            let name = try!(self.name());
            let ty = try!(self.optional_type());
            params.push(PartialParameter { name: name, ty: ty });
            if *self.peek() == TComma {
                self.next();
            } else if *self.peek() != TBar {
                return weld_err!("Expected ',' or '|'")
            }
        }
        try!(self.consume(TBar));
        let body = try!(self.expr());
        Ok(expr_box(Lambda(params, body)))
    }

    /// Parse a sum expression with terms separated by + and - (for operator precedence).
    fn sum_expr(&mut self) -> WeldResult<Box<PartialExpr>> {
        let mut res = try!(self.product_expr());
        while *self.peek() == TPlus || *self.peek() == TMinus {
            let token = self.next();
            let right = try!(self.product_expr());
            if *token == TPlus {
                res = expr_box(BinOp(Add, res, right))
            } else {
                res = expr_box(BinOp(Subtract, res, right))
            }
        }
        Ok(res)
    }

    /// Parse a product expression with terms separated by * and /.
    fn product_expr(&mut self) -> WeldResult<Box<PartialExpr>> {
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

    /// Parse a terminal expression at the bottom of the precedence chain.
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
                try!(self.consume(TCloseBracket));
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
                try!(self.consume(TCloseBrace));
                Ok(expr_box(MakeStruct(exprs)))
            }

            ref other => weld_err!("Expected expression but got '{}'", other)
        }
    }

    /// Parse a name starting at the current input position.
    fn name(&mut self) -> WeldResult<Symbol> {
        match *self.next() {
            TIdent(ref name) => Ok(name.clone()),
            ref other => weld_err!("Expected identifier but got '{}'", other)
        }
    }

    /// Optionally parse a type annotation such as ": i32" and return the result as a PartialType;
    /// gives Unknown if there is no type annotation at the current position. 
    fn optional_type(&mut self) -> WeldResult<PartialType> {
        if *self.peek() == TColon {
            try!(self.consume(TColon));
            self.partial_type() 
        } else {
            Ok(Unknown)
        }
    }

    /// Parse a PartialType starting at the current input position.  
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
                        try!(self.consume(TOpenBracket));
                        let elem_type = try!(self.partial_type());
                        try!(self.consume(TCloseBracket));
                        Ok(Vector(Box::new(elem_type)))
                    }

                    "appender" => {
                        try!(self.consume(TOpenBracket));
                        let elem_type = try!(self.partial_type());
                        try!(self.consume(TCloseBracket));
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
                try!(self.consume(TCloseBrace));
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

    let e = parse_expr("|a, b:i32| a+b").unwrap();
    assert_eq!(print_expr(&e), "|a,b|(a+b)");

    assert!(parse_expr("10 * * 2").is_err());

    let t = parse_type("{i32, vec[vec[?]], ?}").unwrap();
    assert_eq!(print_type(&t), "{i32,vec[vec[?]],?}");

    let t = parse_type("{}").unwrap();
    assert_eq!(print_type(&t), "{}");
}