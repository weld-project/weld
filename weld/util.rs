
extern crate libc;
extern crate fnv;

use std::cmp::max;

use super::ast::*;
use super::ast::ExprKind::*;


/// Utility struct that can track and generate unique IDs and symbols for use in an expression.
/// Each SymbolGenerator tracks the maximum ID used for every symbol name, and can be used to
/// create new symbols with the same name but a unique ID.
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
            let id = id_map.entry(symbol.name.clone()).or_insert(0);
            *id = max(*id, symbol.id);
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
        Symbol {
            name: name.to_owned(),
            id: *id,
        }
    }

    /// Return the next ID that will be given to a symbol with the given string name.
    pub fn next_id(&self, name: &str) -> i32 {
        match self.id_map.get(name) {
            Some(id) => id + 1,
            None => 0,
        }
    }
}

/// Utility struct to generate string IDs with a given prefix.
pub struct IdGenerator {
    prefix: String,
    next_id: i32,
}

impl IdGenerator {
    /// Initialize an IdGenerator that will begin counting up from 0.
    pub fn new(prefix: &str) -> IdGenerator {
        IdGenerator {
            prefix: String::from(prefix),
            next_id: 0,
        }
    }

    /// Generate a new ID.
    pub fn next(&mut self) -> String {
        let res = format!("{}{}", self.prefix, self.next_id);
        self.next_id += 1;
        res
    }
}
