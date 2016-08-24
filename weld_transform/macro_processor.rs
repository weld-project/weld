//! Applies macros to an expression or program, yielding a final `PartialExpr`.
//!
//! Caveats / TODOs:
//! - These macros are not yet hygienic if they define identifiers, though that could be done
//!   by generating new Symbols for those.
//! - Macros that reuse a parameter twice have its expansion appear twice, instead of assigning
//!   it to a temporary as would happen with function application.

use std::vec::Vec;
use std::collections::HashMap;

use weld_ast::*;
use weld_ast::ExprKind::*;
use weld_ast::program::*;
use weld_ast::partial_types::*;
use weld_error::*;
use weld_parser::parse_macros;

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

/// Sanitizes macros by assigning unique symbol names to each symbol defined
/// in the macro.
fn sanitize_expr(e: &mut PartialExpr, symid: &mut usize) {
    let mut names: HashMap<Symbol, Symbol> = HashMap::new();
    match e.kind {
        Let(ref name, _, _) => {
            let mut new_name = String::new();
            new_name.push_str(name.as_ref());
            new_name.push_str("__Macro");
            new_name.push_str(symid.to_string().as_ref());
            names.insert(new_name, name.clone());
        }
        Lambda(ref params, _) => {
            for p in params {
                let mut new_name = String::new();
                new_name.push_str(p.name.as_ref());
                new_name.push_str("__Macro");
                new_name.push_str(symid.to_string().as_ref());
                names.insert(new_name, p.name.clone());
            }
        }
        _ => {}
    }

    // If we found a match, substitute with new name.
    for (old, new) in names {
        let new_expr = PartialExpr {
            ty: e.ty.clone(),
            kind: Ident(new),
        };
        e.substitute(&old, &new_expr);
    }

    for c in e.children_mut() {
        sanitize_expr(c, symid);
    }
}

fn apply_macros(expr: &mut PartialExpr, macros: &HashMap<Symbol, &Macro>) -> WeldResult<bool> {
    let mut new_expr = None;
    let mut symid = 0;
    if let Apply(ref func, ref params) = expr.kind {
        if let Ident(ref name) = func.kind {
            if let Some(mac) = macros.get(name) {
                let mut new = mac.body.clone();
                if params.len() != mac.parameters.len() {
                    return weld_err!("Wrong number of parameters for macro {}", mac.name);
                }
                // Sanitize the macro by replacing names defined in the body
                // with new ones.
                sanitize_expr(&mut new, &mut symid);
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
