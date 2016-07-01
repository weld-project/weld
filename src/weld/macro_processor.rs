//! Applies macros to an expression or program, yielding a final `PartialExpr`.

use std::vec::Vec;
use std::collections::HashMap;

use super::ast::*;
use super::error::*;
use super::program::*;
use super::partial_types::*;

pub fn process_program(program: &Program) -> WeldResult<PartialExpr> {
    process_expression(&program.entry_point, &program.macros)
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

    Ok(expr.clone())
}