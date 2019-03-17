//! Defines the `Uniquify` trait on expressions.
//!
//! This trait makes symbol names unique in the AST.

use super::ast::ExprKind::*;
use super::ast::*;
use crate::error::*;

use std::collections::hash_map::Entry;

use fnv;

#[cfg(test)]
use crate::tests::*;

/// A trait that uniquifies symbol names in-place.
pub trait Uniquify {
    /// Modifies an expression to make symbol names unique.
    ///
    /// Returns an error if an undefined symbol is encountered.
    fn uniquify(&mut self) -> WeldResult<()>;
}

impl Uniquify for Expr {
    fn uniquify(&mut self) -> WeldResult<()> {
        uniquify_helper(self, &mut SymbolStack::new())
    }
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
                let name = ent.key().name();
                let id = ent.get().last().cloned().ok_or_else(|| {
                    WeldCompileError::new(format!("Symbol {} is out of scope", &sym))
                })?;
                Ok(Symbol::new(name, id))
            }
            _ => compile_err!("Undefined symbol {}", sym),
        }
    }

    /// Push a new symbol onto the stack, assigning it a unique name. This enters a new scope for
    /// the name. The symbol can be retrieved with `symbol()`.
    fn push_symbol(&mut self, sym: Symbol) {
        let stack_entry = self.stack.entry(sym.clone()).or_insert_with(Vec::new);
        let next_entry = self.next_unique_symbol.entry(sym.name()).or_insert(-1);
        *next_entry = if sym.id() > *next_entry {
            sym.id()
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
            }
            _ => compile_err!("Attempting to pop undefined symbol {}", sym),
        }
    }
}

/// The main helper function for uniquify, which uses `SymbolStack` to track scope and assign
/// unique names to each symbol. The prerequisite is that each symbol has `id = 0`.
fn uniquify_helper(expr: &mut Expr, symbol_stack: &mut SymbolStack) -> WeldResult<()> {
    match expr.kind {
        // First, handle expressions which define *new* symbols - Let and Lambda
        Lambda {
            ref mut params,
            ref mut body,
        } => {
            // Update the parameter of the lambda with new names.
            let original_params = params.clone();
            for param in params.iter_mut() {
                let sym = &mut param.name;
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
        Let {
            ref mut name,
            ref mut value,
            ref mut body,
        } => {
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

#[test]
fn parse_and_print_uniquified_expressions() {
    let mut e = parse_expr("let a = 2; a").unwrap();
    let _ = e.uniquify();
    assert_eq!(print_expr_without_indent(&e).as_str(), "(let a=(2);a)");

    // Redefine a symbol.
    let mut e = parse_expr("let a = 2; let a = 3; a").unwrap();
    let _ = e.uniquify();
    assert_eq!(
        print_expr_without_indent(&e).as_str(),
        "(let a=(2);(let a__1=(3);a__1))"
    );

    // Make sure Let values aren't renamed.
    let mut e = parse_expr("let a = 2; let a = a+1; a").unwrap();
    let _ = e.uniquify();
    assert_eq!(
        print_expr_without_indent(&e).as_str(),
        "(let a=(2);(let a__1=((a+1));a__1))"
    );

    // Lambdas and proper scoping.
    let mut e = parse_expr("let a = 2; (|a,b|a+b)(1,2) + a").unwrap();
    let _ = e.uniquify();
    assert_eq!(
        print_expr_without_indent(&e).as_str(),
        "(let a=(2);((|a__1,b|(a__1+b))(1,2)+a))"
    );

    // Lambdas and Lets
    let mut e = parse_expr("let b = for([1], appender[i32], |b,i,e| merge(b, e)); b").unwrap();
    let _ = e.uniquify();
    assert_eq!(
        print_expr_without_indent(&e).as_str(),
        "(let b__1=(for([1],appender[i32],|b,i,e|merge(b,e)));b__1)"
    );
}
