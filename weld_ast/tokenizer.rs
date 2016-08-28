//! Breaks strings into Weld tokens for use in the parser.

use std::fmt;
use std::str::FromStr;
use std::vec::Vec;

use regex::Regex;

use weld_error::*;

#[derive(Clone,Debug,PartialEq)]
pub enum Token {
    TI32Literal(i32),
    TBoolLiteral(bool),
    TIdent(String),
    TIf,
    TFor,
    TLet,
    TMacro,
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
    TEqual,
    TBar,           // |
    TDot,
    TColon,
    TSemicolon,
    TQuestion,
    TEndOfInput
}

/// Break up a string into tokens.
pub fn tokenize(input: &str) -> WeldResult<Vec<Token>> {
    lazy_static! {
        // Regular expression for splitting up tokens.
        static ref TOKEN_RE: Regex = Regex::new(
            r"[A_Za-z0-9_]+|[-+/*,=()[\]{}|&\.:;?]|\S+").unwrap();

        // Regular expressions for various types of tokens. 
        static ref KEYWORD_RE: Regex = Regex::new(r"if|for|let|true|false|macro").unwrap();
        static ref IDENT_RE: Regex = Regex::new(r"^[A_Za-z_][A_Za-z0-9_]*$").unwrap();
        static ref I32_RE: Regex = Regex::new(r"^[0-9]+$").unwrap();
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
                "macro" => TMacro,
                "true" => TBoolLiteral(true),
                "false" => TBoolLiteral(false),
                _ => return weld_err!("Invalid input token: {}", text)
            });
        } else if IDENT_RE.is_match(text) {
            tokens.push(TIdent(text.to_string()));
        } else if I32_RE.is_match(text) {
            match i32::from_str(text) {
                Ok(value) => tokens.push(TI32Literal(value)),
                Err(_) => return weld_err!("Invalid i32 literal: {}", text)
            }
        } else {
            tokens.push(match text {
                "+" => TPlus,
                "-" => TMinus,
                "*" => TTimes,
                "/" => TDivide,
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
            TI32Literal(ref value) => write!(f, "{}", value),
            TBoolLiteral(ref value) => write!(f, "{}", value),
            TIdent(ref value) => write!(f, "{}", value),
            ref other => write!(f, "{}", match *other {
                TI32Literal(_) => "",
                TBoolLiteral(_) => "",
                TIdent(_) => "",
                TIf => "if",
                TFor => "for",
                TLet => "let",
                TMacro => "macro",
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
                TEqual => "=",
                TBar => "|",
                TDot => ".",
                TColon => ":",
                TSemicolon => ";",
                TQuestion => "?",
                TEndOfInput => "<END>"
            })
        }
    }
}

#[test]
fn basic_tokenize() {
    use self::Token::*;

    assert_eq!(tokenize("a for 23 + z0").unwrap(),
        vec![TIdent("a".into()), TFor, TI32Literal(23), TPlus, TIdent("z0".into()), TEndOfInput]);

    assert!(tokenize("0a").is_err());
    assert!(tokenize("#").is_err());
}