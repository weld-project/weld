//! Hash-based symbol agnostic AST comparison.
//! 
//! Developers should use the `expr_hash` module when checking whether an AST has changed, or in
//! other scenarios that would otherwise require cloning the AST. In general, computing a hash,
//! modifying the AST, and then comparing the hash of the old AST to the new one is much faster
//! than cloning the AST and then checking if the tree was mutated using
//! `compare_ignoring_symbols`.

extern crate fnv;

use super::ast::*;
use super::ast::ExprKind::*;
use super::ast::LiteralKind::*;
use error::*;

use std::collections::hash_map::Entry;
use std::hash::{Hash, Hasher};

use std::fmt;

/// A trait that implements symbol-agnostic hashing.
pub trait HashIgnoringSymbols {
    /// Hash an AST ignoring symbol names.
    ///
    /// This method is useful for comparing two ASTs for structural equality, e.g., to check if an
    /// optimization pass changed the AST modulo changing symbol names. Comparing using a hash
    /// value is generally faster than cloning a tree, changing it, and then checking if it
    /// changed. This method may return an error if it encounters an undefined symbol.
    fn hash_ignoring_symbols(&self) -> WeldResult<u64>;
}

impl HashIgnoringSymbols for Expr {
    fn hash_ignoring_symbols(&self) -> WeldResult<u64> {
        Ok(ExprHash::from(self)?.value())
    }
}

/// A signature which uniquely represents an Expression in a concise manner.
struct ExprHash {
    hasher: fnv::FnvHasher,
}

impl fmt::Debug for ExprHash {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ExprHash({})", self.hasher.finish())
    }
}

impl ExprHash {
    /// Recurisvely computes signatures for an expression and each of its subexpressions.
    /// The precise symbol names defined within the expression are ignored.
    fn from_expr<'a>(&mut self, expr: &'a Expr,
                                    symbol_positions: &mut fnv::FnvHashMap<&'a Symbol, Vec<i32>>,
                                    max_id: &mut i32) -> WeldResult<()> {
        // In expressions that define new symbols, subexpressions will be handled in the match.
        let mut finished_subexpressions = false;
        // Hash the type.
        expr.ty.hash(&mut self.hasher);
        // Hash the name, which represents the kind of the expression.
        expr.kind.name().hash(&mut self.hasher);
        // Hash the kind.
        match expr.kind {
            Literal(ref kind) => {
                match *kind {
                    BoolLiteral(v) => v.hash(&mut self.hasher),
                    I8Literal(v) => v.hash(&mut self.hasher),
                    I16Literal(v) => v.hash(&mut self.hasher),
                    I32Literal(v) => v.hash(&mut self.hasher),
                    I64Literal(v) => v.hash(&mut self.hasher),
                    U8Literal(v) => v.hash(&mut self.hasher),
                    U16Literal(v) => v.hash(&mut self.hasher),
                    U32Literal(v) => v.hash(&mut self.hasher),
                    U64Literal(v) => v.hash(&mut self.hasher),
                    F32Literal(v) => v.hash(&mut self.hasher),
                    F64Literal(v) => v.hash(&mut self.hasher),
                    StringLiteral(ref v) => v.hash(&mut self.hasher)
                }
            }
            Ident(ref sym) => {
                // We track symbols to disambiguate redefinitions, but also to ignore the actual
                // textual symbol name. By hashing a number representing the symbol, expressions
                // which are structurally the same but have different names will produce the same
                // hash value.
                match symbol_positions.entry(sym) {
                    Entry::Occupied(ref ent) => {
                        ent.get().hash(&mut self.hasher);
                    },
                    _ => {
                        return compile_err!("Undefined symbol {}", sym);
                    }
                }
            }
            BinOp {ref kind, .. } => {
                kind.hash(&mut self.hasher);
            }
            UnaryOp {ref kind, .. } => {
                kind.hash(&mut self.hasher);
            }
            Cast { ref kind, .. } =>  {
                kind.hash(&mut self.hasher);
            }
            GetField { ref index, .. } => {
                index.hash(&mut self.hasher);
            }
            Let { ref name, ref value, ref body } => {
                // Do the value before pushing onto the symbol staack.
                self.from_expr(value, symbol_positions, max_id)?;
                {
                    let entry = symbol_positions.entry(name).or_insert(Vec::new());
                    entry.push(*max_id);
                    *max_id += 1;
                } // brackets to end the borrow.
                self.from_expr(body, symbol_positions, max_id)?;
                // pop the stack.
                let entry = symbol_positions.entry(name).or_insert(Vec::new());
                let _ = entry.pop();
                finished_subexpressions = true;
            }
            Lambda { ref params, ref body } => {
                // Push the stack for each param.
                for param in params.iter() {
                    let entry = symbol_positions.entry(&param.name).or_insert(Vec::new());
                    entry.push(*max_id);
                    *max_id += 1;
                }
                self.from_expr(body, symbol_positions, max_id)?;
                // Pop the stack.
                for param in params.iter() {
                    let entry = symbol_positions.entry(&param.name).or_insert(Vec::new());
                    entry.pop();
                }
                finished_subexpressions = true;
            }
            CUDF { ref sym_name, ref return_ty, .. } => {
                sym_name.hash(&mut self.hasher);
                return_ty.hash(&mut self.hasher);
            }
            Deserialize  { ref value_ty, .. } => {
                value_ty.hash(&mut self.hasher);
            }
            For { ref iters, .. } => {
                for iter in iters.iter() {
                    iter.kind.hash(&mut self.hasher);
                }
            }
            // Other expressions (listed explicitly so we don't forget to add new ones). If the
            // expression doesn't have a non-Expr field, it goes here.
            Negate(_) | Not(_) | Broadcast(_) | Serialize(_) | ToVec{ .. } | MakeStruct { .. } | MakeVector { .. } |
                Zip { .. } | Length { .. } | Lookup { .. } | OptLookup { .. } | KeyExists { .. } |
                Slice { .. } | Sort { .. } | If { .. } | Iterate { .. } | Select { .. } | Apply { .. } |
                NewBuilder(_) | Merge { .. } | Res { .. } => {}
        }
        if !finished_subexpressions {
            for child in expr.children() {
                self.from_expr(child, symbol_positions, max_id)?;
            }
        }
        Ok(())
    }

    /// Return a numeric value for this signature.
    pub fn value(&self) -> u64 {
        return self.hasher.finish() 
    }

    /// Create a signature from an expression.
    pub fn from(expr: &Expr) -> WeldResult<ExprHash> {
        let mut sig = ExprHash { hasher: fnv::FnvHasher::default() };
        let mut symbol_positions = fnv::FnvHashMap::default();
        let mut max_id = 0;
        sig.from_expr(expr, &mut symbol_positions, &mut max_id)?;
        Ok(sig)
    }
}

// For comparing two signatures.
impl PartialEq for ExprHash {
    fn eq(&self, other: &ExprHash) -> bool {
        self.value() == other.value()
    }
}

#[cfg(test)]
use syntax::parser::*;

#[test]
fn test_compare_same() {
    let ref a = parse_expr("|| let a = 1; let b = 1; a").unwrap();
    let ref b = parse_expr("|| let a = 1; let b = 1; a").unwrap();
    assert_eq!(ExprHash::from(a).unwrap(), ExprHash::from(b).unwrap());
}

#[test]
fn test_compare_different_symbols() {
    let ref a = parse_expr("|| let a = 1; let b = 1; a").unwrap();
    let ref b = parse_expr("|| let c = 1; let d = 1; c").unwrap();
    assert_eq!(ExprHash::from(a).unwrap(), ExprHash::from(b).unwrap());
}

#[test]
fn test_compare_different_symbols_ne() {
    let ref a = parse_expr("|| let a = 1; let b = 1; a").unwrap();
    let ref b = parse_expr("|| let c = 1; let d = 1; d").unwrap();
    assert!(ExprHash::from(a).unwrap() != ExprHash::from(b).unwrap());
}


#[test]
fn test_lambda() {
    let ref a = parse_expr("|a: i32| let a = 1; let b = 1; a").unwrap();
    let ref b = parse_expr("|a: i32| let a = 1; let c = 1; a").unwrap();
    assert_eq!(ExprHash::from(a).unwrap(), ExprHash::from(b).unwrap());
}
