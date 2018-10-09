//! Implements common subexpression elimination.

use ast::*;
use ast::constructors::*;
use ast::ExprKind::*;

use util::{join, SymbolGenerator};

use std::collections::{HashMap, HashSet};

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

        cse.remove_common_subexpressions(expr, bindings);

        // Convert the bindings map to map from Expr -> Symbol to Symbol -> Expr.
        let old_length = bindings.len();
        let ref mut bindings = bindings.drain()
            .map(|(k, v)| (v, k))
            .collect::<HashMap<Symbol, Expr>>();
        assert_eq!(bindings.len(), old_length);

        // Print Bindings.
        let mut debug = String::new();
        for (k, v) in bindings.iter() {
            debug.push_str(&format!("{} -> {}\n", k, v.pretty_print()));
        }
        trace!("bindings: {}", debug);
        trace!("Expression after assigning cse symbols: {}", expr.pretty_print());

        let ref dependencies = cse.build_dependencies(bindings);

        // Print Bindings.
        let mut debug = String::new();
        for (k, v) in dependencies.iter() {
            debug.push_str(&format!("{} -> {}\n",
                                    k,
                                    join("[", ",", "]", v.iter().map(|e| e.to_string()))
                                    )
                           );
        }
        trace!("dependencies: {}", debug);

        cse.generate_bindings(expr, bindings, dependencies, &mut HashSet::new(), &mut vec![]);
        
        trace!("After CSE: {}", expr.pretty_print());
    }

    /// Removes common subexpressions and remove bindings.
    ///
    /// This method updates the bindings map to hold identifier to expression information for each
    /// symbol in the expression. The resulting expression is transformed so common subexpressions
    /// are replaced with a common identifier and `Let` statements are removed.
    fn remove_common_subexpressions(&mut self,
                                    expr: &mut Expr,
                                    bindings: &mut HashMap<Expr, Symbol>) {

        trace!("Remove Common Subexpressions called on {}", expr.pretty_print());

        let mut cse_symbols = HashSet::new();

        expr.transform_up(&mut |ref mut e| {
            let mut taken = None;
            match e.kind {
                // literals and identifiers are "unit" expressions that are not eliminated.
                // Lambdas should never be assigned to an identifier.
                Literal(_) | Ident(_) | Lambda { .. } => (),
                Let { ref name, ref mut value, ref mut body } => {
                    
                    // Recursive call to the value.
                    self.remove_common_subexpressions(value, bindings);

                    // Let expressions represent "already CSE'd" values, so we can track them. 
                    cse_symbols.insert(name.clone());
                    let taken_value = value.take();
                    bindings.insert(*taken_value, name.clone());
                    taken = Some(body.take());
                }
                _ => {
                    if !bindings.contains_key(e) {
                        let name = self.sym_gen.new_symbol("cse");
                        cse_symbols.insert(name.clone());
                        bindings.insert(e.clone(), name);
                    }

                    // Replace the expression with its CSE name.
                    let name = bindings.get(e).cloned().unwrap();
                    **e = Expr {
                        ty: e.ty.clone(),
                        kind: Ident(name),
                        annotations: Annotations::new(),
                    };
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

    /// Returns a map with a list of dependencies for each binding in the given map.
    ///
    /// Dependencies track identifiers whose definition must be inserted *before* a given binding.
    /// As an example, consider the following binding:
    ///
    /// ```
    /// cse2 -> cse1 + cse0
    /// ```
    ///
    /// The variable `cse2` depends on `cse1` and `cse0`, so the returned map will include the
    /// value `cse2 -> [cse1, cse0]`.
    ///
    /// Only identifiers that are keys in the bindings map are considered dependencies here. Other
    /// identifiers (i.e., function parameters) are already defined in the expression.
    fn build_dependencies(&mut self, bindings: &HashMap<Symbol, Expr>) -> HashMap<Symbol, HashSet<Symbol>> {
        let mut deps = HashMap::new();
        for (sym, expr) in bindings.iter() {
            let mut symbol_deps = HashSet::new();
            expr.traverse(&mut |ref e| {
                if let Ident(ref name) = e.kind {
                    // We only want other bindings to be dependencies -- other identifiers (i.e.,
                    // function parameters) already exist in the expression.
                    if bindings.contains_key(name) {
                        symbol_deps.insert(name.clone());
                    }
                }
            });
            deps.insert(sym.clone(), symbol_deps);
        }

        deps
    }

    /// Generates bindings in the expression.
    ///
    /// `expr` is the expression to generate the bindings in.
    /// `bindings` maps symbols to their values. Expressions from here are drained and added back
    /// into the expression.
    /// `dependencies` lists the dependent symbols of each symbol in `bindings`.
    /// `generated` tracks which bindings have been inserted into `expr` already. 
    /// `stack` is an ordered list of bindings to be generated in the current lambda.
    fn generate_bindings(&mut self,
                         expr: &mut Expr,
                         bindings: &mut HashMap<Symbol, Expr>,
                         dependencies: &HashMap<Symbol, HashSet<Symbol>>,
                         generated: &mut HashSet<Symbol>,
                         stack: &mut Vec<Vec<(Symbol, Expr)>>) {

        expr.transform_and_continue(&mut |ref mut e| {
            match e.kind {
                Lambda { ref mut body, .. } => {
                    stack.push(vec![]);
                    trace!("pushed to stack");

                    trace!("Processing lambda with body {}", body.pretty_print());

                    let generated_before = generated.clone();
                    self.generate_bindings(body, bindings, dependencies, generated, stack);

                    let binding_list = stack.pop().unwrap();
                    trace!("popped stack");
                    let mut prev = *body.take(); // XXX what is this?
                    for (sym, expr) in binding_list.into_iter().rev() {
                        if !generated_before.contains(&sym) {
                            generated.remove(&sym);
                        }
                        prev = let_expr(sym, expr, prev).unwrap();
                    }

                    **body = prev;
                    trace!("Replaced lambda body with {}", body.pretty_print());
                    (None, false)
                }
                If { ref mut cond, ref mut on_true, ref mut on_false } => {

                    // Generate bindings for the condition, since we won't recursive down into
                    // subexpressions.
                    self.generate_bindings(cond, bindings, dependencies, generated, stack);

                    // We want to keep definitions of If/Else statements within the branch target
                    // to prevent moving expressions out from behind a branch condition.
                    stack.push(vec![]);
                    trace!("pushed to stack");
                    trace!("Processing If on_true {}", on_true.pretty_print());
                    let generated_before = generated.clone();
                    self.generate_bindings(on_true, bindings, dependencies, generated, stack);
                    let binding_list = stack.pop().unwrap();
                    trace!("popped stack");
                    let mut prev = *on_true.take(); // XXX what is this?
                    for (sym, expr) in binding_list.into_iter().rev() {
                        if !generated_before.contains(&sym) {
                            generated.remove(&sym);
                        }
                        prev = let_expr(sym, expr, prev).unwrap();
                    }
                    **on_true = prev;
                    trace!("Replaced If on_true with {}", on_true.pretty_print());

                    stack.push(vec![]);
                    trace!("pushed to stack");
                    trace!("Processing If on_false {}", on_false.pretty_print());
                    let generated_before = generated.clone();
                    self.generate_bindings(on_false, bindings, dependencies, generated, stack);
                    let binding_list = stack.pop().unwrap();
                    trace!("popped stack");
                    let mut prev = *on_false.take(); // XXX what is this?
                    for (sym, expr) in binding_list.into_iter().rev() {
                        if !generated_before.contains(&sym) {
                            generated.remove(&sym);
                        }
                        prev = let_expr(sym, expr, prev).unwrap();
                    }
                    **on_false = prev;
                    trace!("Replaced If on_false with {}", on_false.pretty_print());
                    (None, false)
                }
                Ident(ref mut name) if !generated.contains(name) && dependencies.contains_key(name) => {
                    self.add_dependencies(name, bindings, dependencies, generated, stack);
                    (None, false)
                }
                _ => (None, true),
            }
        });
    }

    /// Add the dependencies of the given symbol to the given stack.
    ///
    /// Dependencies are added recursively.
    fn add_dependencies(&mut self,
                            sym: &Symbol,
                            bindings: &mut HashMap<Symbol, Expr>,
                            dependencies: &HashMap<Symbol, HashSet<Symbol>>,
                            generated: &mut HashSet<Symbol>,
                            stack: &mut Vec<Vec<(Symbol, Expr)>>) {
        if !generated.contains(sym) {
            let mut expr = bindings.get(sym).cloned().unwrap();
            generated.insert(sym.clone());

            self.generate_bindings(&mut expr, bindings, dependencies, generated, stack);

            trace!("Added binding {} -> {}", &sym, expr.pretty_print());
            let binding_list = stack.last_mut().unwrap();
            binding_list.push((sym.clone(), expr));
        }
    }
}

/// Implements the CSE transform. 
pub fn common_subexpression_elimination(expr: &mut Expr) {
    Cse::apply(expr)
}
