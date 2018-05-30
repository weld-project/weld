//! Transform to uniquify symbols in an expession.

use ast::*;
use ast::ExprKind::*;
use error::*;

use std::collections::hash_map::Entry;

extern crate fnv;

/// Modifies symbol names so each symbol is unique in the AST.  Returns an error if an undeclared
/// symbol appears in the program.
pub fn uniquify(expr: &mut Expr) -> WeldResult<()> {
    uniquify_helper(expr, &mut SymbolStack::new())
}

/// A stack which keeps track of unique variables names and scoping information for symbols.
struct SymbolStack {
    // The symbol stack.
    stack: fnv::FnvHashMap<Symbol, Vec<i32>>,
    // The next unique ID to assign to this name.
    next_unique_symbol: fnv::FnvHashMap<String, i32>,
}

impl SymbolStack {
    fn new() -> SymbolStack {
        SymbolStack {
            stack: fnv::FnvHashMap::default(),
            next_unique_symbol: fnv::FnvHashMap::default(),
        }
    }

    /// Returns the symbol in the current scope for the given symbol, or an error if the symbol is
    /// undefined.
    fn symbol(&mut self, sym: Symbol) -> WeldResult<Symbol> {
        match self.stack.entry(sym.clone()) {
            Entry::Occupied(ref ent) => {
                let name = ent.key().name.as_str();
                let id = ent.get()
                    .last()
                    .map(|v| *v)
                    .ok_or(WeldCompileError::new(format!("Symbol {} is out of scope", &sym)))?;
                Ok(Symbol::new(name, id))
            }
            _ => compile_err!("Undefined symbol {}", sym),
        }
    }


    /// Push a new symbol onto the stack, assigning it a unique name. This enters a new scope for
    /// the name. The symbol can be retrieved with `symbol()`.
    fn push_symbol(&mut self, sym: Symbol) {
        let stack_entry = self.stack.entry(sym.clone()).or_insert(Vec::new());
        let next_entry = self.next_unique_symbol.entry(sym.name).or_insert(-1);
        *next_entry = if sym.id > *next_entry {
            sym.id
        } else {
           *next_entry + 1 
        };
        stack_entry.push(*next_entry);
    }

    /// Pop a symbol from the stack.
    fn pop_symbol(&mut self, sym: Symbol) -> WeldResult<()> {
        match self.stack.entry(sym.clone()) {
            Entry::Occupied(mut ent) => {
                ent.get_mut().pop();
                Ok(())
            },
            _ => compile_err!("Attempting to pop undefined symbol {}", sym)
        }
    }
}

/// The main helper function for uniquify, which uses `SymbolStack` to track scope and assign
/// unique names to each symbol. The prerequisite is that each symbol has `id = 0`.
fn uniquify_helper(expr: &mut Expr, symbol_stack: &mut SymbolStack) -> WeldResult<()> {
    match expr.kind {
        // First, handle expressions which define *new* symbols - Let and Lambda
        Lambda {ref mut params, ref mut body} => {
            // Update the parameter of the lambda with new names.
            let original_params = params.clone();
            for param in params.iter_mut() {
                let ref mut sym = param.name;
                symbol_stack.push_symbol(sym.clone());
                *sym = symbol_stack.symbol(sym.clone())?;
            }

            // Then, uniquify the lambda using the newly pushed symbols.
            uniquify_helper(body, symbol_stack)?;
            
            // Finally, pop off the symbol names since they are out of scope now.
            for param in original_params {
                symbol_stack.pop_symbol(param.name)?;
            }
        }
        Let {ref mut name, ref mut value, ref mut body} => {
            // First, uniquify the value *without* the updated stack, since the Let hasn't defined
            // the symbol yet.
            uniquify_helper(value, symbol_stack)?;

            // Now, push the Let's symbol name.
            symbol_stack.push_symbol(name.clone());
            let original_name = name.clone();
            *name = symbol_stack.symbol(name.clone())?;

            // uniquify the body with the new scope.
            uniquify_helper(body, symbol_stack)?;

            // Pop off the scope.
            symbol_stack.pop_symbol(original_name)?;
        }
        // Now handle identifiers, which are changed to reflect new symbols.
        Ident(ref mut sym) => {
            *sym = symbol_stack.symbol(sym.clone())?;
        }
        // For all other expressions, call uniquify_helper on the children.
        _ => {
            for child in expr.children_mut() {
                uniquify_helper(child, symbol_stack)?;
            }
        }
    }
    Ok(())
}
