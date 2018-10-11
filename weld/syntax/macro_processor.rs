//! Applies macros to an expression or program, yielding a final `Expr`.
//!
//! Caveats:
//! - Macros that reuse a parameter twice have its expansion appear twice, instead of assigning
//!   it to a temporary as would happen with function application.

use std::collections::HashMap;
use std::vec::Vec;

use ast::*;
use ast::ExprKind::*;
use super::parser::*;
use super::program::*;
use error::*;
use util::SymbolGenerator;

#[cfg(test)]
use tests::print_expr_without_indent;

const MAX_MACRO_DEPTH: i32 = 30;

// thread_local instead of lazy_static because Rc on Symbol isn't Send.
thread_local! {
    /// Standard macros loaded once.
    static STANDARD_MACROS: Vec<Macro> = {
        let code = include_str!("../resources/standard_macros.weld");
        parse_macros(code).unwrap()
    };
}

/// Apply macros to a program, including the standard macros built into Weld.
pub fn process_program(program: &Program) -> WeldResult<Expr> {
    let mut all_macros = STANDARD_MACROS.with(|v| v.clone());
    all_macros.extend(program.macros.iter().cloned());
    process_expression(&program.body, &all_macros)
}

/// Apply a specific list of macros to an expression (does not load the standard macros).
pub fn process_expression(expr: &Expr, macros: &Vec<Macro>) -> WeldResult<Expr> {
    let mut macro_map: HashMap<Symbol, &Macro> = HashMap::new();
    for m in macros {
        if macro_map.contains_key(&m.name) {
            return compile_err!("Duplicate macro: {}", m.name);
        } else {
            macro_map.insert(m.name.clone(), &m);
        }
    }

    let mut sym_gen = SymbolGenerator::from_expression(&expr);

    let mut expr = expr.clone();
    for _ in 1..MAX_MACRO_DEPTH {
        if !try!(apply_macros(&mut expr, &macro_map, &mut sym_gen)) {
            return Ok(expr);
        }
    }

    compile_err!("Marco expansion recursed past {} levels", MAX_MACRO_DEPTH)
}

fn apply_macros(expr: &mut Expr,
                macros: &HashMap<Symbol, &Macro>,
                sym_gen: &mut SymbolGenerator)
                -> WeldResult<bool> {
    let mut new_expr = None;
    if let Apply {
               ref func,
               ref params,
           } = expr.kind {
        if let Ident(ref name) = func.kind {
            if let Some(mac) = macros.get(name) {
                let mut new_body = mac.body.clone();
                if params.len() != mac.parameters.len() {
                    return compile_err!("Wrong number of parameters for macro {}", mac.name);
                }
                update_defined_ids(&mut new_body, sym_gen);
                for (name, value) in mac.parameters.iter().zip(params) {
                    new_body.substitute(name, value);
                }
                new_expr = Some(new_body);
            }
        }
    }
    let mut changed = false;
    if let Some(new_expr) = new_expr {
        *expr = new_expr;
        changed = true;
    }
    for c in expr.children_mut() {
        changed |= try!(apply_macros(c, macros, sym_gen));
    }
    Ok(changed)
}

fn update_defined_ids(expr: &mut Expr, sym_gen: &mut SymbolGenerator) {
    if let Let {
               name: ref mut sym,
               ref value,
               ref mut body,
           } = expr.kind {
        if sym.id() == 0 {
            let new_sym = sym_gen.new_symbol(&sym.name());
            let new_ident = Expr {
                kind: Ident(new_sym.clone()),
                ty: value.ty.clone(),
                annotations: Annotations::new(),
            };
            body.substitute(sym, &new_ident);
            *sym = Symbol::new(sym.name(), new_sym.id());
        }
    }
    if let Lambda {
               ref mut params,
               ref mut body,
           } = expr.kind {
        for ref mut param in params {
            let sym = &mut param.name;
            if sym.id() == 0 {
                let new_sym = sym_gen.new_symbol(&sym.name());
                let new_ident = Expr {
                    kind: Ident(new_sym.clone()),
                    ty: param.ty.clone(),
                    annotations: Annotations::new(),
                };
                body.substitute(sym, &new_ident);
                *sym = Symbol::new(sym.name(), new_sym.id());
            }
        }
    }
    for c in expr.children_mut() {
        update_defined_ids(c, sym_gen);
    }
}

#[test]
fn basic_macros() {
    let macros = parse_macros("macro foo(a) = a + a;").unwrap();
    let expr = parse_expr("foo(b * b)").unwrap();
    let result = process_expression(&expr, &macros).unwrap();
    assert_eq!(print_expr_without_indent(&result).as_str(), "((b*b)+(b*b))");

    let macros = parse_macros("macro foo(a) = a + a;").unwrap();
    let expr = parse_expr("foo(foo(b))").unwrap();
    let result = process_expression(&expr, &macros).unwrap();
    assert_eq!(print_expr_without_indent(&result).as_str(), "((b+b)+(b+b))");

    // Macro whose parameter symbol is also used as an argument inside
    let macros = parse_macros("macro foo(a) = a + a;").unwrap();
    let expr = parse_expr("foo(a * a)").unwrap();
    let result = process_expression(&expr, &macros).unwrap();
    assert_eq!(print_expr_without_indent(&result).as_str(), "((a*a)+(a*a))");

    // Multiple macros
    let macros = parse_macros("macro foo(a) = a + a; macro bar(a, b) = a * b;").unwrap();
    let expr = parse_expr("foo(bar(a, b))").unwrap();
    let result = process_expression(&expr, &macros).unwrap();
    assert_eq!(print_expr_without_indent(&result).as_str(), "((a*b)+(a*b))");
}

#[test]
fn macros_introducing_symbols() {
    // If the parameter passed to the macro uses a symbol that we define in its body (x here),
    // we should replace that with a new symbol.
    let macros = parse_macros("macro adder(a) = |x| x+a;").unwrap();
    let expr = parse_expr("adder(x)").unwrap();
    let result = process_expression(&expr, &macros).unwrap();
    assert_eq!(print_expr_without_indent(&result).as_str(), "|x__1:?|(x__1+x)");

    // Same case as above except we define a symbol in a Let instead of Lambda.
    let macros = parse_macros("macro twice(a) = (let x = a; x+x);").unwrap();
    let expr = parse_expr("twice(x)").unwrap();
    let result = process_expression(&expr, &macros).unwrap();
    assert_eq!(print_expr_without_indent(&result).as_str(),
               "(let x__1=(x);(x__1+x__1))");

    // On the other hand, if x is not used in the parameter, keep its ID as is.
    let macros = parse_macros("macro adder(a) = |x| x+a;").unwrap();
    let expr = parse_expr("adder(b)").unwrap();
    let result = process_expression(&expr, &macros).unwrap();
    assert_eq!(print_expr_without_indent(&result).as_str(), "|x:?|(x+b)");

    // Same case as above except we define a symbol in a Let instead of Lambda.
    let macros = parse_macros("macro twice(a) = (let x = a; x+x);").unwrap();
    let expr = parse_expr("twice(b)").unwrap();
    let result = process_expression(&expr, &macros).unwrap();
    assert_eq!(print_expr_without_indent(&result).as_str(),
               "(let x=(b);(x+x))");

    // If a symbol is used multiple times during macro substitution, replace it with distinct IDs.
    let macros = parse_macros("macro adder(a) = |x| x+a;").unwrap();
    let expr = parse_expr("adder(x+adder(x)(1))").unwrap();
    let result = process_expression(&expr, &macros).unwrap();
    assert_eq!(print_expr_without_indent(&result).as_str(),
               "|x__1:?|(x__1+(x+(|x__2|(x__2+x))(1)))");

    // Similar case with multiple macros.
    let macros = parse_macros("macro adder(a)=|x|x+a; macro twice(a)=(let x=a; x+x);").unwrap();
    let expr = parse_expr("adder(twice(x))").unwrap();
    let result = process_expression(&expr, &macros).unwrap();
    assert_eq!(print_expr_without_indent(&result).as_str(),
               "|x__1:?|(x__1+(let x__2=(x);(x__2+x__2)))");
}

#[test]
fn standard_macros() {
    // Check that the standard macros file is loaded
    let program = parse_program("map([1,2,3], |a|a+1)").unwrap();
    let result = process_program(&program).unwrap();
    assert_eq!(print_expr_without_indent(&result).as_str(),
               "result(for([1,2,3],appender[?],|b,i,x|merge(b,(|a|(a+1))(x))))");
}
