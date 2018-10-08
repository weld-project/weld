//! Implements common subexpression elimination.

use ast::*;
use ast::ExprKind::*;
use util::SymbolGenerator;

use std::collections::HashMap;
use std::collections::hash_map::Entry;

/// A trait that generates a typed identifier expression.
trait NewIdentifier {
    /// Return an identifier with the given type and name.
    fn identifier(ty: Type, name: Symbol) -> Self;
}

impl NewIdentifier for Expr {
    fn identifier(ty: Type, name: Symbol) -> Expr {
        Expr {
            ty: ty,
            kind: Ident(name),
            annotations: Annotations::new(),
        }
    }
}

/// State for the CSE transformation.
#[derive(Debug)]
struct Cse {
    /// Maps a subexpression to its identifier.
    ///
    /// Identifiers are lazily instantiated upon their first use.
    subexprs: HashMap<Expr, Option<Expr>>,
    /// Tracks symbols.
    sym_gen: SymbolGenerator,
}

impl Cse {
    /// Apply the CSE transformation on the given expression.
    pub fn apply(expr: &mut Expr) {
        let mut cse = Cse {
            subexprs: HashMap::default(),
            sym_gen: SymbolGenerator::from_expression(expr),
        };
        cse.transform(expr);
    }

    fn transform(&mut self, expr: &mut Expr) {
        use self::NewIdentifier;
        expr.transform_up(&mut |ref mut e| {
            match e.kind {
                // literals and identifiers are "unit" expressions that are not eliminated.
                Literal(_) | Ident(_) | Lambda { .. } => (),
                _ => {
                    // XXX Can this clone be avoided...
                    match self.subexprs.entry(e.clone()) {
                        Entry::Occupied(ref mut ent) => {
                            // The visited sub-expression has been seen before. We can replace it
                            // with the symbol in the table.
                            
                            // Create an identifier if necessary.
                            if ent.get().is_none() {
                                let name = self.sym_gen.new_symbol("cse");
                                let sym = Expr::identifier(e.ty.clone(), name);
                                *ent.get_mut() =  Some(sym);
                            }
                            **e = ent.get().clone().unwrap();
                        }
                        Entry::Vacant(mut ent) => {
                            // This expression tree has been seen for the first time -- register it
                            // in the table. We insert None here and lazily generate a new
                            // identifier the first time the identifier is used.
                            ent.insert(None);
                        }
                    }
                }
            };
            None
        });
        debug!("replaced expression: {}", expr.pretty_print());
        debug!("{:?}", self);
    }
}

/// Implements the CSE transform. 
pub fn common_subexpression_elimination(expr: &mut Expr) {
    Cse::apply(expr)
}
