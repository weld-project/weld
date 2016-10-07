//! Applies macros to an expression or program, yielding a final `PartialExpr`.
//!
//! Caveats:
//! - Macros that reuse a parameter twice have its expansion appear twice, instead of assigning
//!   it to a temporary as would happen with function application.

use std::collections::HashMap;
use std::vec::Vec;

use super::ast::*;
use super::ast::ExprKind::*;
use super::program::*;
use super::parser::*;
use super::partial_types::*;
use super::error::*;
use super::util::IdGenerator;

#[cfg(test)] use super::pretty_print::*;

const MAX_MACRO_DEPTH: i32 = 30;

lazy_static! {
    static ref STANDARD_MACROS: Vec<Macro> = {
        let code = include_str!("resources/standard_macros.weld");
        parse_macros(code).unwrap()
    };
}

/// Apply macros to a program, including the standard macros built into Weld.
pub fn process_program(program: &Program) -> WeldResult<PartialExpr> {
    let mut all_macros = STANDARD_MACROS.clone();
    all_macros.extend(program.macros.iter().cloned());
    process_expression(&program.body, &all_macros)
}

/// Apply a specific list of macros to an expression (does not load the standard macros).
pub fn process_expression(expr: &PartialExpr, macros: &Vec<Macro>) -> WeldResult<PartialExpr> {
    let mut macro_map: HashMap<Symbol, &Macro> = HashMap::new();
    for m in macros {
        if macro_map.contains_key(&m.name) {
            return weld_err!("Duplicate macro: {}", m.name);
        } else {
            macro_map.insert(m.name.clone(), &m);
        }
    }

    let mut id_gen = IdGenerator::from_expression(&expr);

    let mut expr = expr.clone();
    for _ in 1..MAX_MACRO_DEPTH {
        if !try!(apply_macros(&mut expr, &macro_map, &mut id_gen)) {
            return Ok(expr)
        }
    }

    weld_err!("Marco expansion recursed past {} levels", MAX_MACRO_DEPTH)
}

fn apply_macros(
    expr: &mut PartialExpr,
    macros: &HashMap<Symbol, &Macro>,
    id_gen: &mut IdGenerator
) -> WeldResult<bool> {
    let mut new_expr = None;
    if let Apply(ref func, ref params) = expr.kind {
        if let Ident(ref name) = func.kind {
            if let Some(mac) = macros.get(name) {
                let mut new_body = mac.body.clone();
                if params.len() != mac.parameters.len() {
                    return weld_err!("Wrong number of parameters for macro {}", mac.name);
                }
                update_defined_ids(&mut new_body, id_gen);
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
        changed |= try!(apply_macros(c, macros, id_gen));
    }
    Ok(changed)
}

fn update_defined_ids(expr: &mut PartialExpr, id_gen: &mut IdGenerator) {
    if let Let(ref mut sym, ref value, ref mut body) = expr.kind {
        if sym.id == 0 {
            let new_sym = id_gen.new_symbol(&sym.name);
            let new_ident = PartialExpr { kind: Ident(new_sym.clone()), ty: value.ty.clone() };
            body.substitute(sym, &new_ident);
            sym.id = new_sym.id;
        }
    }
    if let Lambda(ref mut params, ref mut body) = expr.kind {
        for ref mut param in params {
            let sym = &mut param.name;
            if sym.id == 0 {
                let new_sym = id_gen.new_symbol(&sym.name);
                let new_ident = PartialExpr { kind: Ident(new_sym.clone()), ty: param.ty.clone() };
                body.substitute(sym, &new_ident);
                sym.id = new_sym.id;
            }
        }
    }
    for c in expr.children_mut() {
        update_defined_ids(c, id_gen);
    }
}

#[test]
fn basic_macros() {
    let macros = parse_macros("macro foo(a) = a + a;").unwrap();
    let expr = parse_expr("foo(b * b)").unwrap();
    let result = process_expression(&expr, &macros).unwrap();
    assert_eq!(print_expr(&result).as_str(), "((b*b)+(b*b))");

    let macros = parse_macros("macro foo(a) = a + a;").unwrap();
    let expr = parse_expr("foo(foo(b))").unwrap();
    let result = process_expression(&expr, &macros).unwrap();
    assert_eq!(print_expr(&result).as_str(), "((b+b)+(b+b))");

    // Macro whose parameter symbol is also used as an argument inside
    let macros = parse_macros("macro foo(a) = a + a;").unwrap();
    let expr = parse_expr("foo(a * a)").unwrap();
    let result = process_expression(&expr, &macros).unwrap();
    assert_eq!(print_expr(&result).as_str(), "((a*a)+(a*a))");

    // Multiple macros
    let macros = parse_macros("macro foo(a) = a + a; macro bar(a, b) = a * b;").unwrap();
    let expr = parse_expr("foo(bar(a, b))").unwrap();
    let result = process_expression(&expr, &macros).unwrap();
    assert_eq!(print_expr(&result).as_str(), "((a*b)+(a*b))");
}

#[test]
fn macros_introducing_symbols() {
    // If the parameter passed to the macro uses a symbol that we define in its body (x here),
    // we should replace that with a new symbol.
    let macros = parse_macros("macro adder(a) = |x| x+a;").unwrap();
    let expr = parse_expr("adder(x)").unwrap();
    let result = process_expression(&expr, &macros).unwrap();
    assert_eq!(print_expr(&result).as_str(), "|x#1|(x#1+x)");

    // Same case as above except we define a symbol in a Let instead of Lambda.
    let macros = parse_macros("macro twice(a) = (let x = a; x+x);").unwrap();
    let expr = parse_expr("twice(x)").unwrap();
    let result = process_expression(&expr, &macros).unwrap();
    assert_eq!(print_expr(&result).as_str(), "(let x#1=(x);(x#1+x#1))");

    // On the other hand, if x is not used in the parameter, keep its ID as is.
    let macros = parse_macros("macro adder(a) = |x| x+a;").unwrap();
    let expr = parse_expr("adder(b)").unwrap();
    let result = process_expression(&expr, &macros).unwrap();
    assert_eq!(print_expr(&result).as_str(), "|x|(x+b)");

    // Same case as above except we define a symbol in a Let instead of Lambda.
    let macros = parse_macros("macro twice(a) = (let x = a; x+x);").unwrap();
    let expr = parse_expr("twice(b)").unwrap();
    let result = process_expression(&expr, &macros).unwrap();
    assert_eq!(print_expr(&result).as_str(), "(let x=(b);(x+x))");

    // If a symbol is used multiple times during macro substitution, replace it with distinct IDs.
    let macros = parse_macros("macro adder(a) = |x| x+a;").unwrap();
    let expr = parse_expr("adder(x+adder(x)(1))").unwrap();
    let result = process_expression(&expr, &macros).unwrap();
    assert_eq!(print_expr(&result).as_str(), "|x#1|(x#1+(x+(|x#2|(x#2+x))(1)))");

    // Similar case with multiple macros.
    let macros = parse_macros("macro adder(a)=|x|x+a; macro twice(a)=(let x=a; x+x);").unwrap();
    let expr = parse_expr("adder(twice(x))").unwrap();
    let result = process_expression(&expr, &macros).unwrap();
    assert_eq!(print_expr(&result).as_str(), "|x#1|(x#1+(let x#2=(x);(x#2+x#2)))");
}