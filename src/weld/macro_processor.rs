//! Applies macros to an expression or program, yielding a final `PartialExpr`.

use std::vec::Vec;
use std::collections::HashMap;

use super::ast::*;
use super::ast::ExprKind::*;
use super::error::*;
use super::program::*;
use super::partial_types::*;

const MAX_MACRO_DEPTH: i32 = 30;

pub fn process_program(program: &Program) -> WeldResult<PartialExpr> {
    process_expression(&program.body, &program.macros)
}

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
        if !try!(apply_recursive(&mut expr, &macro_map)) {
            return Ok(expr)
        }
    }

    weld_err!("Marco expansion recursed past {} levels", MAX_MACRO_DEPTH)
}

fn apply_recursive(
        expr: &mut PartialExpr,
        macro_map: &HashMap<Symbol, &Macro>)
        -> WeldResult<bool> {
    let mut new_expr = None;
    if let Apply(ref func, ref params) = expr.kind {
        if let Ident(ref name) = func.kind {
            if let Some(mac) = macro_map.get(name) {
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
        changed |= try!(apply_recursive(c, macro_map));
    }
    Ok(changed)
}