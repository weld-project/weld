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

use error::*;

#[derive(Clone,Debug,PartialEq)]
pub enum Token {
    TI32Literal(i32),
    TI64Literal(i64),
    TF32Literal(f32),
    TF64Literal(f64),
    TI16Literal(i16),
    TI8Literal(i8),
    TBoolLiteral(bool),
    TStringLiteral(String),
    TIdent(String),
    TIf,
    TIterate,
    TFor,
    TMerge,
    TResult,
    TLet,
    TMacro,
    TI8,
    TI16,
    TI32,
    TI64,
    TU8,
    TU16,
    TU32,
    TU64,
    TF32,
    TF64,
    TBool,
    TVec,
    TDict,
    TZip,
    TScalarIter,
    TSimdIter,
    TFringeIter,
    TNdIter,
    TRangeIter,
    TLen,
    TLookup,
    TOptLookup,
    TKeyExists,
    TSlice,
    TSort,
    TExp,
    TSin,
    TCos,
    TTan,
    TASin,
    TACos,
    TATan,
    TSinh,
    TCosh,
    TTanh,
    TSimd,
    TSelect,
    TBroadcast,
    TSerialize,
    TDeserialize,
    TLog,
    TErf,
    TSqrt,
    TCUDF,
    TAppender,
    TMerger,
    TDictMerger,
    TGroupMerger,
    TVecMerger,
    TToVec,
    TOpenParen, // (
    TCloseParen, // )
    TOpenBracket, // [
    TCloseBracket, // ]
    TOpenBrace, // {
    TCloseBrace, // }
    TComma,
    TPlus,
    TMinus,
    TTimes,
    TDivide,
    TModulo,
    TEqual,
    TBar, // |
    TAtMark, // @
    TDot,
    TColon,
    TSemicolon,
    TQuestion,
    TBang,
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
    TMax,
    TMin,
    TPow,
    TEndOfInput,
}

impl Token {
    /// Returns whether this token requires a space preceding it when printing a
    /// token stream.
    pub fn requires_space(&self) -> bool {
        use super::tokenizer::Token::*;
        match *self {
            TOpenParen |
                TCloseParen |
                TOpenBracket |
                TCloseBracket |
                TOpenBrace |
                TCloseBrace |
                TComma |
                TPlus |
                TMinus |
                TTimes |
                TDivide |
                TModulo |
                TEqual |
                TBar |
                TAtMark |
                TDot |
                TColon |
                TSemicolon |
                TQuestion |
                TBang |
                TEqualEqual |
                TNotEqual |
                TLessThanOrEqual |
                TGreaterThanOrEqual |
                TLessThan |
                TGreaterThan |
                TLogicalAnd |
                TLogicalOr |
                TBitwiseAnd |
                TXor |
                TMax |
                TMin |
                TPow |
                TEndOfInput => false,
            _ => true
        }
    }
}

/// Break up a string into tokens.
pub fn tokenize(input: &str) -> WeldResult<Vec<Token>> {
    lazy_static! {
        // Regular expression for splitting up tokens.
        static ref TOKEN_RE: Regex = Regex::new(concat!(
            "(?m)#.*$|",
            r#"[0-9]+\.[0-9]+([eE]-?[0-9]+)?[fF]?|[0-9]+[eE]-?[0-9]+[fF]?|"[^"]*"|"#,
            r#"[A-Za-z0-9$_]+|==|!=|>=|<=|&&|\|\||[-+/*%,=()[\]{}|@&\.:;?!&\|^<>]|\S+"#
        )).unwrap();

        // Regular expressions for various types of tokens.
        static ref KEYWORD_RE: Regex = Regex::new(
            "^(if|for|zip|len|lookup|optlookup|keyexists|slice|sort|exp|sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|\
             log|erf|sqrt|simd|select|broadcast|serialize|deserialize|\
             iterate|cudf|simditer|fringeiter|rangeiter|nditer|iter|merge|result|let|true|false|macro|\
             i8|i16|i32|i64|u8|u16|u32|u64|f32|f64|bool|vec|dict|appender|merger|vecmerger|\
             dictmerger|groupmerger|tovec|min|max|pow)$").unwrap();

        static ref COMMENT_RE: Regex = Regex::new("#.*$").unwrap();
        static ref STRLIT_RE: Regex = Regex::new(r#""[^"]*""#).unwrap();
        static ref IDENT_RE: Regex = Regex::new(r"^[A-Za-z$_][A-Za-z0-9$_]*$").unwrap();

        static ref I8_BASE_10_RE: Regex = Regex::new(r"^[0-9]+[cC]$").unwrap();
        static ref I8_BASE_2_RE: Regex = Regex::new(r"^0b[0-1]+[cC]$").unwrap();
        static ref I8_BASE_16_RE: Regex = Regex::new(r"^0x[0-9a-fA-F]+[cC]$").unwrap();

	static ref I16_BASE_10_RE: Regex = Regex::new(r"[0-9]+si$").unwrap();
	static ref I16_BASE_2_RE: Regex = Regex::new(r"^0b[0-1]+si$").unwrap();
        static ref I16_BASE_16_RE: Regex = Regex::new(r"^0x[0-9a-fA-F]+si$").unwrap();

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
        if COMMENT_RE.is_match(text) {
            // Do nothing - skips the token.
        } else if KEYWORD_RE.is_match(text) {
            tokens.push(match text {
                            "if" => TIf,
                            "iterate" => TIterate,
                            "let" => TLet,
                            "for" => TFor,
                            "merge" => TMerge,
                            "result" => TResult,
                            "macro" => TMacro,
                            "i8" => TI8,
                            "i16" => TI16,
                            "i32" => TI32,
                            "i64" => TI64,
                            "u8" => TU8,
                            "u16" => TU16,
                            "u32" => TU32,
                            "u64" => TU64,
                            "f32" => TF32,
                            "f64" => TF64,
                            "bool" => TBool,
                            "vec" => TVec,
                            "dict" => TDict,
                            "appender" => TAppender,
                            "merger" => TMerger,
                            "dictmerger" => TDictMerger,
                            "groupmerger" => TGroupMerger,
                            "vecmerger" => TVecMerger,
                            "tovec" => TToVec,
                            "zip" => TZip,
                            "iter" => TScalarIter,
                            "simditer" => TSimdIter,
                            "fringeiter" => TFringeIter,
                            "nditer" => TNdIter,
                            "rangeiter" => TRangeIter,
                            "len" => TLen,
                            "lookup" => TLookup,
                            "optlookup" => TOptLookup,
                            "keyexists" => TKeyExists,
                            "slice" => TSlice,
                            "sort" => TSort,
                            "exp" => TExp,
                            "sin" => TSin,
                            "cos" => TCos,
                            "tan" => TTan,
                            "asin" => TASin,
                            "acos" => TACos,
                            "atan" => TATan,
                            "sinh" => TSinh,
                            "cosh" => TCosh,
                            "tanh" => TTanh,
                            "log" => TLog,
                            "erf" => TErf,
                            "sqrt" => TSqrt,
                            "cudf" => TCUDF,
                            "simd" => TSimd,
                            "select" => TSelect,
                            "broadcast" => TBroadcast,
                            "serialize" => TSerialize,
                            "deserialize" => TDeserialize,
                            "true" => TBoolLiteral(true),
                            "false" => TBoolLiteral(false),
                            "min" => TMin,
                            "max" => TMax,
                            "pow" => TPow,
                            _ => return compile_err!("Invalid input token: {}", text),
                        });
        } else if STRLIT_RE.is_match(text) {
            let string = text.trim_matches('"').to_string();
            if !(string.is_ascii()) {
                return compile_err!("Weld strings must be valid ASCII");
            }
            tokens.push(TStringLiteral(string)); // Trim off quotes before tokenizing
        } else if IDENT_RE.is_match(text) {
            tokens.push(TIdent(text.to_string()));
        } else if I8_BASE_10_RE.is_match(text) {
            tokens.push(try!(parse_i8_literal(text, 10)))
        } else if I8_BASE_2_RE.is_match(text) {
            tokens.push(try!(parse_i8_literal(text, 2)))
        } else if I8_BASE_16_RE.is_match(text) {
            tokens.push(try!(parse_i8_literal(text, 16)))
        } else if I16_BASE_10_RE.is_match(text) {
            tokens.push(try!(parse_i16_literal(text, 10)))
        } else if I16_BASE_2_RE.is_match(text) {
            tokens.push(try!(parse_i16_literal(text, 2)))
        } else if I16_BASE_16_RE.is_match(text) {
            tokens.push(try!(parse_i16_literal(text, 16)))
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
            match f32::from_str(&text[..text.len() - 1]) {
                Ok(value) => tokens.push(Token::TF32Literal(value)),
                Err(_) => return compile_err!("Invalid f32 literal: {}", text),
            }
        } else if F64_RE.is_match(text) {
            match f64::from_str(text) {
                Ok(value) => tokens.push(Token::TF64Literal(value)),
                Err(_) => return compile_err!("Invalid f64 literal: {}", text),
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
                            "@" => TAtMark,
                            "," => TComma,
                            "=" => TEqual,
                            "." => TDot,
                            ":" => TColon,
                            ";" => TSemicolon,
                            "?" => TQuestion,
                            "!" => TBang,
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
                            _ => return compile_err!("Invalid input token: {}", text),
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
            TI16Literal(ref value) => write!(f, "{}si", value),
            TI8Literal(ref value) => write!(f, "{}C", value),
            TBoolLiteral(ref value) => write!(f, "{}B", value),
            TStringLiteral(ref value) => write!(f, "\"{}\"", value),
            TIdent(ref value) => write!(f, "{}", value),

            // Cases that return fixed strings
            ref other => {
                write!(f, "{}", match *other {
                    // These cases are handled above but repeated here for exhaustive match
                    TI32Literal(_) => "",
                    TI64Literal(_) => "",
                    TF32Literal(_) => "",
                    TF64Literal(_) => "",
                    TI16Literal(_) => "",
                    TI8Literal(_) => "",
                    TBoolLiteral(_) => "",
                    TStringLiteral(_) => "",
                    TIdent(_) => "",
                    // Other cases that return fixed strings
                    TIf => "if",
                    TIterate => "iterate",
                    TFor => "for",
                    TMerge => "merge",
                    TResult => "result",
                    TLet => "let",
                    TMacro => "macro",
                    TI8 => "i8",
                    TI16 => "i16",
                    TI32 => "i32",
                    TI64 => "i64",
                    TU8 => "u8",
                    TU16 => "u16",
                    TU32 => "u32",
                    TU64 => "u64",
                    TF32 => "f32",
                    TF64 => "f64",
                    TBool => "bool",
                    TVec => "vec",
                    TDict => "dict",
                    TAppender => "appender",
                    TMerger => "merger",
                    TDictMerger => "dictmerger",
                    TGroupMerger => "groupmerger",
                    TVecMerger => "vecmerger",
                    TToVec => "tovec",
                    TZip => "zip",
                    TScalarIter => "iter",
                    TSimdIter => "simditer",
                    TFringeIter => "fringeiter",
                    TNdIter => "nditer",
                    TRangeIter => "rangeiter",
                    TLen => "len",
                    TLookup => "lookup",
                    TOptLookup => "optlookup",
                    TKeyExists => "keyexists",
                    TSlice => "slice",
                    TSort => "sort",
                    TExp => "exp",
                    TSin => "sin",
                    TCos => "cos",
                    TTan => "tan",
                    TASin => "asin",
                    TACos => "atan",
                    TATan => "acos",
                    TSinh => "sinh",
                    TCosh => "cosh",
                    TTanh => "tanh",
                    TLog => "log",
                    TErf => "erf",
                    TSqrt => "sqrt",
                    TCUDF => "cudf",
                    TSimd => "simd",
                    TSelect => "select",
                    TBroadcast => "broadcast",
                    TSerialize => "serialize",
                    TDeserialize => "deserialize",
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
                    TAtMark => "@",
                    TDot => ".",
                    TColon => ":",
                    TSemicolon => ";",
                    TQuestion => "?",
                    TBang => "!",
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
                    TMin => "min",
                    TMax => "max",
                    TPow => "pow",
                    TEndOfInput => "<END>",
                })
            }
        }
    }
}

fn parse_i8_literal(input: &str, base: u32) -> WeldResult<Token> {
    let slice = if base == 10 {
        &input[..input.len() - 1]
    } else {
        &input[2..input.len() - 1]
    };
    match i8::from_str_radix(slice, base) {
        Ok(value) => Ok(Token::TI8Literal(value)),
        Err(_) => compile_err!("Invalid i8 literal: {}", input),
    }
}

fn parse_i16_literal(input: &str, base: u32) -> WeldResult<Token> {
    let slice = if base == 10 {
        &input[..input.len() - 2]
    } else {
        &input[2..input.len() - 2]
    };
    match i16::from_str_radix(slice, base) {
        Ok(value) => Ok(Token::TI16Literal(value)),
        Err(_) => compile_err!("Invalid i16 literal: {}", input), 
    }
}

fn parse_i32_literal(input: &str, base: u32) -> WeldResult<Token> {
    let slice = if base == 10 { input } else { &input[2..] };
    match i32::from_str_radix(slice, base) {
        Ok(value) => Ok(Token::TI32Literal(value)),
        Err(_) => compile_err!("Invalid i32 literal: {}", input),
    }
}

fn parse_i64_literal(input: &str, base: u32) -> WeldResult<Token> {
    let slice = if base == 10 {
        &input[..input.len() - 1]
    } else {
        &input[2..input.len() - 1]
    };
    match i64::from_str_radix(slice, base) {
        Ok(value) => Ok(Token::TI64Literal(value)),
        Err(_) => compile_err!("Invalid i32 literal: {}", input),
    }
}

#[test]
fn basic_tokenize() {
    use self::Token::*;

    assert_eq!(tokenize("\"test string\"").unwrap(),
               vec![TStringLiteral("test string".to_string()),
                    TEndOfInput]);
    assert_eq!(tokenize("\"test\" string").unwrap(),
               vec![TStringLiteral("test".to_string()),
                    TIdent("string".into()),
                    TEndOfInput]);
    assert_eq!(tokenize("a for 23 + z0").unwrap(),
               vec![TIdent("a".into()),
                    TFor,
                    TI32Literal(23),
                    TPlus,
                    TIdent("z0".into()),
                    TEndOfInput]);
    assert_eq!(tokenize("groupmerger[i32, i32]").unwrap(),
               vec![TGroupMerger,
                    TOpenBracket,
                    TI32,
                    TComma,
                    TI32,
                    TCloseBracket,
                    TEndOfInput]);

    assert_eq!(tokenize("= == | || & &&").unwrap(),
               vec![TEqual,
                    TEqualEqual,
                    TBar,
                    TLogicalOr,
                    TBitwiseAnd,
                    TLogicalAnd,
                    TEndOfInput]);
    assert_eq!(tokenize("|a:i8| a").unwrap(),
               vec![TBar,
                    TIdent("a".into()),
                    TColon,
                    TI8,
                    TBar,
                    TIdent("a".into()),
                    TEndOfInput]);
    assert_eq!(tokenize("|a:vec[i8]| slice(a, 2L, 3L)").unwrap(),
               vec![TBar,
                    TIdent("a".into()),
                    TColon,
                    TVec,
                    TOpenBracket,
                    TI8,
                    TCloseBracket,
                    TBar,
                    TSlice,
                    TOpenParen,
                    TIdent("a".into()),
                    TComma,
                    TI64Literal(2),
                    TComma,
                    TI64Literal(3),
                    TCloseParen,
                    TEndOfInput]);
    assert_eq!(tokenize("|a:i8| exp(a)").unwrap(),
               vec![TBar,
                    TIdent("a".into()),
                    TColon,
                    TI8,
                    TBar,
                    TExp,
                    TOpenParen,
                    TIdent("a".into()),
                    TCloseParen,
                    TEndOfInput]);
    assert_eq!(tokenize("|a:i8| log(a)").unwrap(),
               vec![TBar,
                    TIdent("a".into()),
                    TColon,
                    TI8,
                    TBar,
                    TLog,
                    TOpenParen,
                    TIdent("a".into()),
                    TCloseParen,
                    TEndOfInput]);
    assert_eq!(tokenize("|a:i8| erf(a)").unwrap(),
               vec![TBar,
                    TIdent("a".into()),
                    TColon,
                    TI8,
                    TBar,
                    TErf,
                    TOpenParen,
                    TIdent("a".into()),
                    TCloseParen,
                    TEndOfInput]);
    assert_eq!(tokenize("|a:i8| sqrt(a)").unwrap(),
               vec![TBar,
                    TIdent("a".into()),
                    TColon,
                    TI8,
                    TBar,
                    TSqrt,
                    TOpenParen,
                    TIdent("a".into()),
                    TCloseParen,
                    TEndOfInput]);
    assert_eq!(tokenize("iffy if").unwrap(),
               vec![TIdent("iffy".into()),
                    TIf,
                    TEndOfInput]);

    assert_eq!(tokenize("keyexists(a, 1)").unwrap(),
               vec![TKeyExists,
                    TOpenParen,
                    TIdent("a".into()),
                    TComma,
                    TI32Literal(1),
                    TCloseParen,
                    TEndOfInput]);
    assert!(tokenize("0a").is_err());

    assert_eq!(tokenize("42si").unwrap(), vec![TI16Literal(42i16), TEndOfInput]);
    assert_eq!(tokenize("0b10").unwrap(), vec![TI32Literal(2), TEndOfInput]);
    assert_eq!(tokenize("0x10").unwrap(),
               vec![TI32Literal(16), TEndOfInput]);

    assert_eq!(tokenize("1e-5f").unwrap(),
               vec![TF32Literal(1e-5f32), TEndOfInput]);
    assert_eq!(tokenize("1e-5").unwrap(),
               vec![TF64Literal(1e-5), TEndOfInput]);
    assert_eq!(tokenize("dictmerger[i32,i32,+] @[]").unwrap(),
               vec![TDictMerger,
                    TOpenBracket,
                    TI32,
                    TComma,
                    TI32,
                    TComma,
                    TPlus,
                    TCloseBracket,
                    TAtMark,
                    TOpenBracket,
                    TCloseBracket,
                    TEndOfInput]);
    assert_eq!(tokenize("sort(a, |x:i32| x)").unwrap(),
               vec![TSort,
                    TOpenParen,
                    TIdent("a".into()),
                    TComma,
                    TBar,
                    TIdent("x".into()),
                    TColon,
                    TI32,
                    TBar,
                    TIdent("x".into()),
                    TCloseParen,
                    TEndOfInput]);
}
