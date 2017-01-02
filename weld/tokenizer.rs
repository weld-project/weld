//! Breaks strings into Weld tokens for use in the parser.
//!
//! This module works by splitting the input string using a regular expression designed to find
//! entire, non-overlapping patterns, such as literals (1e-5), identifiers (var1), and other
//! individual tokens (+, -, $, if, |, etc). The regular expresions for various token types are
//! matched greedily in an order that ensures the "largest" one wins first; for example, the
//! string '1e-5' is parsed as a f64 literal, not as ('1e', '-', '5').

use std::fmt;
use std::str::FromStr;
use std::vec::Vec;

use regex::Regex;

use super::error::*;

#[derive(Clone,Debug,PartialEq)]
pub enum Token {
    TI32Literal(i32),
    TI64Literal(i64),
    TF32Literal(f32),
    TF64Literal(f64),
    TBoolLiteral(bool),
    TIdent(String),
    TIf,
    TFor,
    TMerge,
    TResult,
    TLet,
    TMacro,
    TI32,
    TI64,
    TF32,
    TF64,
    TBool,
    TVec,
    TZip,
    TIter,
    TLen,
    TAppender,
    TOpenParen,     // (
    TCloseParen,    // )
    TOpenBracket,   // [
    TCloseBracket,  // ]
    TOpenBrace,     // {
    TCloseBrace,    // }
    TComma,
    TPlus,
    TMinus,
    TTimes,
    TDivide,
    TModulo,
    TEqual,
    TBar,           // |
    TDot,
    TColon,
    TSemicolon,
    TQuestion,
    TEqualEqual,
    TNotEqual,
    TLessThanOrEqual,
    TGreaterThanOrEqual,
    TLessThan,
    TGreaterThan,
    TLogicalAnd,
    TLogicalOr,
    TBitwiseAnd,
    TXor,
    TEndOfInput
}

/// Break up a string into tokens.
pub fn tokenize(input: &str) -> WeldResult<Vec<Token>> {
    lazy_static! {
        // Regular expression for splitting up tokens.
        static ref TOKEN_RE: Regex = Regex::new(concat!(
            r"[0-9]+\.[0-9]+([eE]-?[0-9]+)?[fF]?|[0-9]+[eE]-?[0-9]+[fF]?|",
            r"[A-Za-z0-9$_]+|==|!=|>=|<=|&&|\|\||[-+/*%,=()[\]{}|&\.:;?&\|^<>]|\S+"
        )).unwrap();

        // Regular expressions for various types of tokens.
        static ref KEYWORD_RE: Regex = Regex::new(
            "if|for|zip|len|iter|merge|result|let|true|false|macro|i32|i64|f32|f64|bool|vec|appender").unwrap();

        static ref IDENT_RE: Regex = Regex::new(r"^[A-Za-z$_][A-Za-z0-9$_]*$").unwrap();

        static ref I32_BASE_10_RE: Regex = Regex::new(r"^[0-9]+$").unwrap();
        static ref I32_BASE_2_RE: Regex = Regex::new(r"^0b[0-1]+$").unwrap();
        static ref I32_BASE_16_RE: Regex = Regex::new(r"^0x[0-9a-fA-F]+$").unwrap();

        static ref I64_BASE_10_RE: Regex = Regex::new(r"^[0-9]+[lL]$").unwrap();
        static ref I64_BASE_2_RE: Regex = Regex::new(r"^0b[0-1]+[lL]$").unwrap();
        static ref I64_BASE_16_RE: Regex = Regex::new(r"^0x[0-9a-fA-F]+[lL]$").unwrap();

        static ref F32_RE: Regex = Regex::new(
            r"[0-9]+\.[0-9]+([eE]-?[0-9]+)?[fF]|[0-9]+([eE]-?[0-9]+)?[fF]").unwrap();

        static ref F64_RE: Regex = Regex::new(
            r"[0-9]+\.[0-9]+([eE]-?[0-9]+)?|[0-9]+[eE]-?[0-9]+").unwrap();
    }

    use self::Token::*;

    let mut tokens: Vec<Token> = Vec::new();

    for cap in TOKEN_RE.captures_iter(input) {
        let text = cap.at(0).unwrap();
        if KEYWORD_RE.is_match(text) {
            tokens.push(match text {
                "if" => TIf,
                "let" => TLet,
                "for" => TFor,
                "merge" => TMerge,
                "result" => TResult,
                "macro" => TMacro,
                "i32" => TI32,
                "i64" => TI64,
                "f32" => TF32,
                "f64" => TF64,
                "bool" => TBool,
                "vec" => TVec,
                "appender" => TAppender,
                "zip" => TZip,
                "iter" => TIter,
                "len" => TLen,
                "true" => TBoolLiteral(true),
                "false" => TBoolLiteral(false),
                _ => return weld_err!("Invalid input token: {}", text)
            });
        } else if IDENT_RE.is_match(text) {
            tokens.push(TIdent(text.to_string()));
        } else if I32_BASE_10_RE.is_match(text) {
            tokens.push(try!(parse_i32_literal(text, 10)))
        } else if I32_BASE_2_RE.is_match(text) {
            tokens.push(try!(parse_i32_literal(text, 2)))
        } else if I32_BASE_16_RE.is_match(text) {
            tokens.push(try!(parse_i32_literal(text, 16)))
        } else if I64_BASE_10_RE.is_match(text) {
            tokens.push(try!(parse_i64_literal(text, 10)))
        } else if I64_BASE_2_RE.is_match(text) {
            tokens.push(try!(parse_i64_literal(text, 2)))
        } else if I64_BASE_16_RE.is_match(text) {
            tokens.push(try!(parse_i64_literal(text, 16)))
        } else if F32_RE.is_match(text) {
            match f32::from_str(&text[..text.len()-1]) {
                Ok(value) => tokens.push(Token::TF32Literal(value)),
                Err(_) => return weld_err!("Invalid f32 literal: {}", text)
            }
        } else if F64_RE.is_match(text) {
            match f64::from_str(text) {
                Ok(value) => tokens.push(Token::TF64Literal(value)),
                Err(_) => return weld_err!("Invalid f32 literal: {}", text)
            }
        } else {
            tokens.push(match text {
                "+" => TPlus,
                "-" => TMinus,
                "*" => TTimes,
                "/" => TDivide,
                "%" => TModulo,
                "(" => TOpenParen,
                ")" => TCloseParen,
                "[" => TOpenBracket,
                "]" => TCloseBracket,
                "{" => TOpenBrace,
                "}" => TCloseBrace,
                "|" => TBar,
                "," => TComma,
                "=" => TEqual,
                "." => TDot,
                ":" => TColon,
                ";" => TSemicolon,
                "?" => TQuestion,
                "==" => TEqualEqual,
                "!=" => TNotEqual,
                "<" => TLessThan,
                ">" => TGreaterThan,
                "<=" => TLessThanOrEqual,
                ">=" => TGreaterThanOrEqual,
                "&&" => TLogicalAnd,
                "||" => TLogicalOr,
                "&" => TBitwiseAnd,
                "^" => TXor,
                _ => return weld_err!("Invalid input token: {}", text)
            });
        }
    }
    tokens.push(TEndOfInput);
    return Ok(tokens);
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::Token::*;
        match *self {
            // Cases that return variable strings
            TI32Literal(ref value) => write!(f, "{}", value),
            TI64Literal(ref value) => write!(f, "{}L", value),
            TF32Literal(ref value) => write!(f, "{}F", value),
            TF64Literal(ref value) => write!(f, "{}", value),  // TODO: force .0?
            TBoolLiteral(ref value) => write!(f, "{}", value),
            TIdent(ref value) => write!(f, "{}", value),

            // Cases that return fixed strings
            ref other => write!(f, "{}", match *other {
                // These cases are handled above but repeated here for exhaustive match
                TI32Literal(_) => "",
                TI64Literal(_) => "",
                TF32Literal(_) => "",
                TF64Literal(_) => "",
                TBoolLiteral(_) => "",
                TIdent(_) => "",
                // Other cases that return fixed strings
                TIf => "if",
                TFor => "for",
                TMerge => "merge",
                TResult => "result",
                TLet => "let",
                TMacro => "macro",
                TI32 => "i32",
                TI64 => "i64",
                TF32 => "f32",
                TF64 => "f64",
                TBool => "bool",
                TVec => "vec",
                TAppender => "appender",
                TZip => "zip",
                TIter => "iter",
                TLen => "len",
                TOpenParen => "(",
                TCloseParen => ")",
                TOpenBracket => "[",
                TCloseBracket => "]",
                TOpenBrace => "{",
                TCloseBrace => "}",
                TComma => ",",
                TPlus => "+",
                TMinus => "-",
                TTimes => "*",
                TDivide => "/",
                TModulo => "%",
                TEqual => "=",
                TBar => "|",
                TDot => ".",
                TColon => ":",
                TSemicolon => ";",
                TQuestion => "?",
                TEqualEqual => "==",
                TNotEqual => "!=",
                TLessThan => "<",
                TGreaterThan => ">",
                TLessThanOrEqual => "<=",
                TGreaterThanOrEqual => ">=",
                TLogicalAnd => "&&",
                TLogicalOr => "||",
                TBitwiseAnd => "&",
                TXor => "^",
                TEndOfInput => "<END>"
            })
        }
    }
}

fn parse_i32_literal(input: &str, base: u32) -> WeldResult<Token> {
    let slice = if base == 10 { input } else { &input[2..] };
    match i32::from_str_radix(slice, base) {
        Ok(value) => Ok(Token::TI32Literal(value)),
        Err(_) => weld_err!("Invalid i32 literal: {}", input)
    }
}

fn parse_i64_literal(input: &str, base: u32) -> WeldResult<Token> {
    let slice = if base == 10 { &input[..input.len()-1] } else { &input[2..input.len()-1] };
    match i64::from_str_radix(slice, base) {
        Ok(value) => Ok(Token::TI64Literal(value)),
        Err(_) => weld_err!("Invalid i32 literal: {}", input)
    }
}

#[test]
fn basic_tokenize() {
    use self::Token::*;

    assert_eq!(tokenize("a for 23 + z0").unwrap(),
        vec![TIdent("a".into()), TFor, TI32Literal(23), TPlus, TIdent("z0".into()), TEndOfInput]);

    assert_eq!(tokenize("= == | || & &&").unwrap(),
        vec![TEqual, TEqualEqual, TBar, TLogicalOr, TBitwiseAnd, TLogicalAnd, TEndOfInput]);

    assert!(tokenize("0a").is_err());
    assert!(tokenize("#").is_err());

    assert_eq!(tokenize("0b10").unwrap(), vec![TI32Literal(2), TEndOfInput]);
    assert_eq!(tokenize("0x10").unwrap(), vec![TI32Literal(16), TEndOfInput]);

    assert_eq!(tokenize("1e-5f").unwrap(), vec![TF32Literal(1e-5f32), TEndOfInput]);
    assert_eq!(tokenize("1e-5").unwrap(), vec![TF64Literal(1e-5), TEndOfInput]);
}
