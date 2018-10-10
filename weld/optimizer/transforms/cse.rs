//! Implements common subexpression elimination.

use ast::*;
use ast::constructors::*;
use ast::ExprKind::*;

use util::SymbolGenerator;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;

/// State for the CSE transformation.
#[derive(Debug)]
struct Cse {
    /// Tracks symbol names.
    sym_gen: SymbolGenerator,
}

impl Cse {
    /// Apply the CSE transformation on the given expression.
    pub fn apply(expr: &mut Expr) {
        let mut cse = Cse {
            sym_gen: SymbolGenerator::from_expression(expr),
        };

        // Maps expressions to symbol names.
        let ref mut bindings = HashMap::new();
        let ref mut aliases = HashMap::new();

        cse.remove_common_subexpressions(expr, bindings, aliases);

        for value in bindings.values() {
            println!("{}", value);
        }

        // Convert the bindings map to map from Expr -> Symbol to Symbol -> Expr.
        let ref mut bindings = bindings.drain()
            .map(|(k, v)| (v, k))
            .collect::<HashMap<Symbol, Expr>>();

        // FIXME: Update bindings list with expressions. This unnecessarily copies the expression -
        // we can CSE this by choosing the "outermost" alias and using it everywhere.
        for (sym, alias_list) in aliases.drain() {
            for alias in alias_list {
                let expr = bindings.get(&sym).cloned().unwrap(); 
                bindings.insert(alias, expr);
            }
        }

        let ref scopes =  cse.build_scope_map(expr, bindings);

        // Debug.
        {
            let mut debug = String::new();
            for (k, v) in bindings.iter() {
                debug.push_str(&format!("{} -> {}\n", k, v.pretty_print()));
            }
            trace!("bindings: {}", debug);
            trace!("Expression after assigning cse symbols: {}", expr.pretty_print());

            let mut debug = String::new();
            for (k, v) in scopes.iter() {
                debug.push_str(&format!("{} -> {}\n", k, v));
            }
            trace!("scopes: {}", debug);
        }

        cse.generate_bindings(expr, bindings, &mut HashSet::new(), &mut vec![], scopes);
        
        trace!("After CSE: {}", expr.pretty_print());
    }

    /// Removes common subexpressions and remove bindings.
    ///
    /// This method updates the bindings map to hold identifier to expression information for each
    /// symbol in the expression. The resulting expression is transformed so common subexpressions
    /// are replaced with a common identifier and `Let` statements are removed.
    ///
    /// To handle the alias issue (see inline comment), we also return a mapping for a symbol name
    /// with its aliases.
    fn remove_common_subexpressions(&mut self,
                                    expr: &mut Expr,
                                    bindings: &mut HashMap<Expr, Symbol>,
                                    aliases: &mut HashMap<Symbol, Vec<Symbol>>) {

        trace!("Remove Common Subexpressions called on {}", expr.pretty_print());

        expr.transform_up(&mut |ref mut e| {
            trace!("Processing {}", e.pretty_print());
            let mut taken = None;
            match e.kind {
                // literals and identifiers are "unit" expressions that are not eliminated.
                //
                // TODO: Struct of newbuilder - should probably ignore anything with nested
                // newbuilder...?
                Literal(_) | Ident(_) | Lambda { .. } | NewBuilder(_) => (),
                Let { ref name, ref mut value, ref mut body } => {
                    // Let expressions represent "already CSE'd" values, so we can track them. 
                    let taken_value = value.take();
                    println!("\t Added symbol {} -> {}", name, taken_value.pretty_print());

                    // ALIAS ISSUE.
                    // We can have cases where a Let statement in the input redefines
                    // the same expression, e.g.,:
                    //
                    // let a = x;
                    // let b = x;
                    //
                    // a + a + b + b
                    //
                    // Since we map expression -> symbol name, the final bindings list will contain
                    // b -> x
                    //
                    // and no reference to a.
                    //
                    // To handle this, we create another map `alises`, that maps the first symbol
                    // with all of its aliases (same value expressions). The alias map is used to
                    // create copies of expressions later in the bindings map.
                    match bindings.entry(*taken_value) {
                        Entry::Vacant(ent) => {
                            ent.insert(name.clone());
                        }
                        Entry::Occupied(ref ent) => {
                            // Alias! We assigned this expression a name already.
                            let sym = ent.get().clone();
                            let alias_list = aliases.entry(sym).or_insert(vec![]);
                            alias_list.push(name.clone());
                        }
                    }
                    taken = Some(body.take());
                }
                _ => {
                    if !bindings.contains_key(e) {
                        let name = self.sym_gen.new_symbol("cse");
                        bindings.insert(e.clone(), name);
                    }

                    // Replace the expression with its CSE name.
                    let name = bindings.get(e).cloned().unwrap();
                    return Some(Expr {
                        ty: e.ty.clone(),
                        kind: Ident(name),
                        annotations: Annotations::new(),
                    });
                }
            };

            // For Let statements, replace the Let with the body. The binding definition
            // is tracked in the bindings map.
            if let Some(body) = taken {
                return Some(*body);
            } else {
                return None;
            }
        });
    }

    /// Build a scope map for each identifier.
    ///
    /// The scope map maps a symbol to the *least deep* scope in an expression tree. For example,
    /// in the code below:
    ///
    /// ```weld
    /// let cse0 = ...; # depth 0
    /// if ( cond,
    ///     let cse0 = ...; # depth 1
    ///     ...
    /// )
    /// ```
    ///
    /// `cse0` appears in two scopes: depth-0 and depth-1. The scope map would indicate depth-0,
    /// since this is the least deep scope.
    fn build_scope_map(&mut self, expr: &Expr, bindings: &HashMap<Symbol, Expr>) -> HashMap<Symbol, isize> {
        let mut scopes = HashMap::new();
        self.build_scope_map_helper(expr, bindings, &mut scopes, -1);
        scopes
    }

    fn build_scope_map_helper(&mut self,
                           expr: &Expr,
                           bindings: &HashMap<Symbol, Expr>, 
                           scope_map: &mut HashMap<Symbol, isize>,
                           current_scope: isize) {

        let mut handled = true;
        match expr.kind {
            If { ref cond, ref on_true, ref on_false } => {
                self.build_scope_map_helper(cond, bindings, scope_map, current_scope);
                self.build_scope_map_helper(on_true, bindings, scope_map, current_scope + 1);
                self.build_scope_map_helper(on_false, bindings, scope_map, current_scope + 1);
            }
            Lambda { ref body, .. } => {
                self.build_scope_map_helper(body, bindings, scope_map, current_scope + 1);
            }
            Ident(ref name) if bindings.contains_key(name) => {
                {
                    let ent = scope_map.entry(name.clone()).or_insert(current_scope);
                    if current_scope < *ent {
                        *ent = current_scope;
                    }
                }
                let expr = bindings.get(name).unwrap();
                self.build_scope_map_helper(expr, bindings, scope_map, current_scope);
            }
            _ => {
                handled = false;
            }
        }

        if !handled {
            for child in expr.children() {
                self.build_scope_map_helper(child, bindings, scope_map, current_scope);
            }
        }
    }

    /// Generates bindings in the expression.
    ///
    /// # Arguments
    ///
    /// * `expr` is the expression to generate the bindings in.
    /// * `bindings` maps symbols to their values. Expressions from here are drained and added back
    /// into the expression.
    /// * `generated` tracks which bindings have been inserted into `expr` already. 
    /// * `stack` is an ordered list of bindings to be generated in the current lambda.
    /// * `scopes` is the *highest* scope that an expression appears in. The expression is only
    /// inserted in this scope and is guaranteed to appear before all references to its symbol. 
    fn generate_bindings(&mut self,
                         expr: &mut Expr,
                         bindings: &mut HashMap<Symbol, Expr>,
                         generated: &mut HashSet<Symbol>,
                         stack: &mut Vec<Vec<(Symbol, Expr)>>,
                         scopes: &HashMap<Symbol, isize>) {

        expr.transform_and_continue(&mut |ref mut e| {
            match e.kind {
                Lambda { ref mut body, .. } => {
                    self.generate_bindings_scoped(body, bindings, generated, stack, scopes);
                    (None, false)
                }
                If { ref mut cond, ref mut on_true, ref mut on_false } => {
                    // Generate bindings for the condition, since we won't recursive down into
                    // subexpressions.
                    self.generate_bindings(cond, bindings, generated, stack, scopes);

                    // We want to keep definitions of If/Else statements within the branch target
                    // to prevent moving expressions out from behind a branch condition.
                    self.generate_bindings_scoped(on_true, bindings, generated, stack, scopes);
                    self.generate_bindings_scoped(on_false, bindings, generated, stack, scopes);
                    (None, false)
                }
                Ident(ref mut sym) if !generated.contains(sym) && bindings.contains_key(sym) => {
                    let mut expr = bindings.get(sym).cloned().unwrap();
                    generated.insert(sym.clone());

                    // Generate the definition of this identifier if it doesn't exist.
                    self.generate_bindings(&mut expr, bindings, generated, stack, scopes);

                    trace!("Added binding {} -> {}", &sym, expr.pretty_print());
                    // We add the generated bindings to the highest scope.
                    let scope = *scopes.get(&sym).unwrap() as usize;
                    let binding_list = stack.get_mut(scope).unwrap();
                    binding_list.push((sym.clone(), expr));
                    (None, false)
                }
                _ => (None, true),
            }
        });
    }

    /// Generates bindings in a new scope.
    ///
    /// This method adds Let expressions within the given expression.
    fn generate_bindings_scoped(&mut self,
                               expr: &mut Expr,
                               bindings: &mut HashMap<Symbol, Expr>,
                               generated: &mut HashSet<Symbol>,
                               stack: &mut Vec<Vec<(Symbol, Expr)>>,
                               scopes: &HashMap<Symbol, isize>) {

        trace!("Processing scoped expression {}", expr.pretty_print());

        stack.push(vec![]);

        // TODO Remove this!
        let generated_before = generated.clone();
        self.generate_bindings(expr, bindings, generated, stack, scopes);

        // Generate the Let expressions for the bindings in this scope.
        let binding_list = stack.pop().unwrap();
        let mut prev = expr.take();
        for (sym, expr) in binding_list.into_iter().rev() {
            if !generated_before.contains(&sym) {
                generated.remove(&sym);
            }
            prev = let_expr(sym, expr, prev).unwrap();
        }

        // Update the expression to contain the Lets.
        *expr = prev;
        trace!("Replaced scoped expression with {}", expr.pretty_print());
    }
}

/// Implements the CSE transform. 
pub fn common_subexpression_elimination(expr: &mut Expr) {
    Cse::apply(expr)
}
