//! Applies macros to an expression or program, yielding a final `PartialExpr`.
//!
//! Caveats / TODOs:
//! - These macros are not yet hygienic if they define identifiers, though that could be done
//!   by generating new Symbols for those.
//! - Macros that reuse a parameter twice have its expansion appear twice, instead of assigning
//!   it to a temporary as would happen with function application.

use std::vec::Vec;
use std::collections::HashMap;

use super::ast::*;
use super::ast::ExprKind::*;
use super::error::*;
use super::grammar::parse_macros;
use super::program::*;
use super::partial_types::*;

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

    let mut expr = expr.clone();
    for _ in 1..MAX_MACRO_DEPTH {
        if !try!(apply_macros(&mut expr, &macro_map)) {
            return Ok(expr)
        }
    }

    weld_err!("Marco expansion recursed past {} levels", MAX_MACRO_DEPTH)
}

fn apply_macros(expr: &mut PartialExpr, macros: &HashMap<Symbol, &Macro>) -> WeldResult<bool> {
    let mut new_expr = None;
    if let Apply(ref func, ref params) = expr.kind {
        if let Ident(ref name) = func.kind {
            if let Some(mac) = macros.get(name) {
                let mut new = mac.body.clone();
                if params.len() != mac.parameters.len() {
                    return weld_err!("Wrong number of parameters for macro {}", mac.name);
                }
                for (name, value) in mac.parameters.iter().zip(params) {
                    new.substitute(name, value);
                }
                new_expr = Some(new);
                //new_expr = Some(PartialExpr { ty: expr.ty.clone(), kind: BoolLiteral(true) });
            }
        }
    }
    let mut changed = false;
    if let Some(new_expr) = new_expr {
        *expr = new_expr;
        changed = true;
    }
    for c in expr.children_mut() {
        changed |= try!(apply_macros(c, macros));
    }
    Ok(changed)
}