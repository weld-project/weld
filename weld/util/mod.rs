//! Compiler utilities.
//!
//! This module contains a number of utilities used throughput the compiler, such as unique symbol
//! and ID generators and a module for measuring compile-time performance of various aspects of the
//! compiler.

extern crate libc;
extern crate fnv;
extern crate time;

use ast::*;
use ast::ExprKind::*;

use std::iter;
use std::cmp::max;

pub mod stats;
pub mod colors;
pub mod dump;

/// Utility struct that can track and generate unique IDs and symbols for use in an expression.
/// Each SymbolGenerator tracks the maximum ID used for every symbol name, and can be used to
/// create new symbols with the same name but a unique ID.
#[derive(Debug,Clone)]
pub struct SymbolGenerator {
    id_map: fnv::FnvHashMap<String, i32>,
}

impl SymbolGenerator {
    /// Initialize a SymbolGenerator with no existing symbols.
    pub fn new() -> SymbolGenerator {
        SymbolGenerator { id_map: fnv::FnvHashMap::default() }
    }

    /// Initialize a SymbolGenerator from all the symbols defined in an expression.
    pub fn from_expression(expr: &Expr) -> SymbolGenerator {
        let mut id_map: fnv::FnvHashMap<String, i32> = fnv::FnvHashMap::default();

        let update_id = |id_map: &mut fnv::FnvHashMap<String, i32>, symbol: &Symbol| {
            let id = id_map.entry(symbol.name().clone()).or_insert(0);
            *id = max(*id, symbol.id());
        };

        expr.traverse(&mut |e| match e.kind {
                               Let { ref name, .. } => update_id(&mut id_map, name),
                               Ident(ref sym) => update_id(&mut id_map, sym),
                               Lambda { ref params, .. } => {
                                   for ref p in params {
                                       update_id(&mut id_map, &p.name);
                                   }
                               }
                               _ => {}
                           });

        SymbolGenerator { id_map: id_map }
    }

    pub fn new_symbol(&mut self, name: &str) -> Symbol {
        let id = self.id_map.entry(name.to_owned()).or_insert(-1);
        *id += 1;
        Symbol::new(name, *id)
    }
}

pub fn join<T: iter::Iterator<Item = String>>(start: &str, sep: &str, end: &str, strings: T) -> String {
    let mut res = String::new();
    res.push_str(start);
    for (i, s) in strings.enumerate() {
        if i > 0 {
            res.push_str(sep);
        }
        res.push_str(&s);
    }
    res.push_str(end);
    res
}

