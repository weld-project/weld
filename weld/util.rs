use std::cmp::max;
use std::collections::HashMap;

use super::ast::*;
use super::ast::ExprKind::*;

/// Utility struct that can track and generate unique IDs and symbols for use in an expression.
/// Each SymbolGenerator tracks the maximum ID used for every symbol name, and can be used to
/// create new symbols with the same name but a unique ID.
pub struct SymbolGenerator {
    id_map: HashMap<String, i32>
}

impl SymbolGenerator {
    /// Initialize anSymbolGenerator from all the symbols defined in an expression.
    pub fn from_expression<T:Clone>(expr: &Expr<T>) -> SymbolGenerator {
        let mut id_map: HashMap<String, i32> = HashMap::new();

        let update_id = |id_map: &mut HashMap<String, i32>, symbol: &Symbol| {
            let id = id_map.entry(symbol.name.clone()).or_insert(0);
            *id = max(*id, symbol.id);
        };

        expr.traverse(&mut |e| {
            match e.kind {
                Let(ref sym, _, _) => update_id(&mut id_map, sym),
                Ident(ref sym) => update_id(&mut id_map, sym),
                Lambda(ref params, _) => {
                    for ref p in params {
                        update_id(&mut id_map, &p.name);
                    }
                },
                _ => {}
            }
        });

        SymbolGenerator { id_map: id_map }
    }

    pub fn new_symbol(&mut self, name: &String) -> Symbol {
        let id = self.id_map.entry(name.clone()).or_insert(-1);
        *id += 1;
        Symbol { name: name.clone(), id: *id }
    }
}

/// Utility struct to generate unique integer IDs.
pub struct IdGenerator {
    next_id: i32
}

impl IdGenerator {
    /// Initialize an IdGenerator that will begin counting up from 0.
    pub fn new() -> IdGenerator {
        IdGenerator { next_id: 0 }
    }

    /// Generate a new ID.
    pub fn next(&mut self) -> i32 {
        let res = self.next_id;
        self.next_id += 1;
        res
    }
}