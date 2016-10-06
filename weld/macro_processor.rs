//! Applies macros to an expression or program, yielding a final `PartialExpr`.
//!
//! Caveats / TODOs:
//! - These macros are not yet hygienic if they define identifiers, though that could be done
//!   by generating new Symbols for those.
//! - Macros that reuse a parameter twice have its expansion appear twice, instead of assigning
//!   it to a temporary as would happen with function application.

use std::collections::HashMap;
use std::vec::Vec;

use super::ast::*;
use super::ast::ExprKind::*;
use super::program::*;
use super::parser::parse_macros;
use super::partial_types::*;
use super::error::*;
use super::util::IdGenerator;

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
            sym.id = new_sym.id;
            let new_ident = PartialExpr { kind: Ident(new_sym), ty: value.ty.clone() };
            body.substitute(sym, &new_ident);
        }
    }
    if let Lambda(ref mut params, ref mut body) = expr.kind {
        for ref mut param in params {
            let sym = &mut param.name;
            if sym.id == 0 {
                let new_sym = id_gen.new_symbol(&sym.name);
                sym.id = new_sym.id;
                let new_ident = PartialExpr { kind: Ident(new_sym), ty: param.ty.clone() };
                body.substitute(sym, &new_ident);
            }
        }
    }
    for c in expr.children_mut() {
        update_defined_ids(c, id_gen);
    }
}