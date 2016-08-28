use std::str::FromStr;
use std::vec::Vec;
use regex::Regex;

use weld_error::*;

#[derive(Clone,Debug,PartialEq)]
pub enum Token {
    TI32Lit(i32),
    TIdent(String),
    TIf,
    TFor,
    TLet,
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
    TBar,           // |
    TEndOfInput
}

/// Break up a string into tokens.
pub fn tokenize(input: &str) -> WeldResult<Vec<Token>> {
    lazy_static! {
        // Regular expression for splitting up tokens.
        static ref TOKEN_RE: Regex = Regex::new(r"[A_Za-z0-9_.]+|[-+/*,()[\]]|\S+").unwrap();

        // Regular expressions for various types of tokens. 
        static ref KEYWORD_RE: Regex = Regex::new(r"if|for|let").unwrap();
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
                _ => return weld_err!("Invalid input token: {}", text)
            });
        } else if IDENT_RE.is_match(text) {
            tokens.push(TIdent(text.to_string()));
        } else if I32_RE.is_match(text) {
            match i32::from_str(text) {
                Ok(value) => tokens.push(TI32Lit(value)),
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
                _ => return weld_err!("Invalid input token: {}", text)
            });
        }
    }

    tokens.push(TEndOfInput);

    return Ok(tokens);
}

#[test]
fn basic_tokenize() {
    use self::Token::*;

    assert_eq!(tokenize("a for 23 + z0").unwrap(),
        vec![TIdent("a".into()), TFor, TI32Lit(23), TPlus, TIdent("z0".into()), TEndOfInput]);

    assert!(tokenize("0a").is_err());
    assert!(tokenize("#").is_err());
}