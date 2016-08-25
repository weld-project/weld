//! Applies macros to an expression or program, yielding a final `PartialExpr`.
//!
//! Caveats / TODOs:
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

/// Performs the work of sanitizing a name by generating a new name,
/// replacing the old one with it, and adding the names into the map
/// so the substitution can be applied to the expression's children.
fn sanitize_name(replace: &mut String,
                 id: &mut usize,
                 ty: PartialType,
                 name_map: &mut HashMap<Symbol, (Symbol, PartialType)>) {
    let mut new_name = String::new();
    let old_name = replace.clone();
    // Generate a new name.
    new_name.push_str("__M");
    new_name.push_str(id.to_string().as_ref());
    new_name.push_str("_");
    new_name.push_str(old_name.as_ref());
    *id += 1;
    // Replace the old name with the new one.
    replace.clear();
    replace.push_str(new_name.as_ref());
    name_map.insert(old_name, (new_name, ty));
}

/// Sanitizes expressions by assigning unique symbol names to each symbol
/// defined in the them.
fn sanitize_expr(e: &mut PartialExpr, symid: &mut usize) {
    let mut names: HashMap<Symbol, (Symbol, PartialType)> = HashMap::new();
    match e.kind {
        Let(ref mut name, _, _) => {
            sanitize_name(name, symid, e.ty.clone(), &mut names);
        }
        Lambda(ref mut params, _) => {
            for p in params {
                sanitize_name(&mut p.name, symid, e.ty.clone(), &mut names);
            }
        }
        _ => ()
    }
    // Replace identifiers in children.
    for (old, new) in names {
        let new_expr = PartialExpr {
            ty: new.1,
            kind: Ident(new.0),
        };
        e.substitute(&old, &new_expr);
    }
    // Apply on children to find nested definitions.
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
