//! Implements common subexpression elimination.
//!
//! Common subexpression elimination (CSE) eliminates redundant work by assigning common
//! subexpressions to a `let` expression. In this module, we refer to the symbol/value pair of a
//! `let` expression as a "binding", and the site of the `let` expression as a "site".
//!
//! The CSE algorithm works in two high level steps: _subexpression elimination_ and  _binding
//! generation_. The algorithm assumes symbol names in the expression are unique (the transform
//! internally uses the uniquify pass).
//!
//! # 1. Subexpression Elimination
//!
//! In subexpression elimination, first, we create a hash table that maps each expression to a
//! symbol name. The hash table is created by traversing the expression tree bottom-up (from leaf
//! to root), so sub-expressions are added to the table before parent expressions. When processing
//! an expression, we check if its exists in the hash table, and if it does, we replace it with the
//! corresponding symbol.
//!
//! The output of this step is a modified expression tree where each expression is assigned an
//! identifier, as well as the bindings table that maps each sub-expression to its symbol name.
//! Note that at this stage, the expression tree itself _does not_ contain the identifier
//! definitions (i.e., there are no `let` statements -- only unbound identifiers).
//!
//! ## Considerations
//!
//! The subexpression elimination step does not eliminate redundant occurrances of the
//! following expression kinds:
//!
//! * Literals
//! * Identifiers
//! * Expressions with Builder types
//!
//! Builder expressions may have side effects (e.g., memory allocations), so we don't eliminate
//! them. In practice, since builders are linear types, we can only have common sub-expressions
//! involving the NewBuilder expression.
//!
//! # 2. Binding Generation
//!
//! In the binding generation phase, we need to insert definitions for each identifier in the
//! expression produced by subexpression elimination. A variable may be used multiple times, and it
//! may also be used in many different sites (e.g., within a For loop's lambda as well as within
//! the top level function), so there are potentially many sites where the binding can be inserted.
//! We also do not want to hoist out expressions computed behind a branch. Finally, identifiers
//! whose values themselves depend on other identifiers must be generated in the correct order.
//! Note that in our context, a _site_ is a site where we can insert a binding. Lambdas and branch
//! targets both create new sites.
//!
//! As an example, consider the following Weld program, annotated with sites:
//!
//! ```weld
//! |x: i32, w: vec[i32], v: vec[i32], z: vec[i32]|      | 
//!     lookup(w, 1) +                                   |
//!     if(x > 0,                                        |
//!         lookup(v, 0) + lookup(v, 0) + lookup(w, 1),  | |_______site [0, 1]
//!         lookup(z, 0)                                 | |_______site [0, 2]
//!     )                                                |_________site [0]
//! ```
//!
//! There are several expressions here: the lookup on `w`, the lookup on `v`, and the lookup on `z`
//! are the most notable. Of these, `lookup(w, 1)` and `lookup(v, 0)` are redundant, but `lookup(v,
//! 0)` should occur after the `if` condition check (i.e., the let statement for it should occur in
//! the true condition of the branch).
//!
//! The bindings generator takes as input an expression that looks as follows:
//!
//! ```weld
//! |x: i32, w: vec[i32], v: vec[i32], z: vec[i32]|
//! cse7
//! ```
//!
//! and a bindings list (not that this is the sub-expression -> symbol map generated during the
//! first step with keys/values flipped):
//!
//! ```weld
//! cse0 -> lookup(w, 1)
//! cse1 -> lookup(v, 0)
//! cse2 -> lookup(z, 0)
//! cse3 -> x > 0
//! cse4 -> cse1 + cse1
//! cse5 -> cse1 + cse0
//! cse6 -> if(cse3, cse5, cse2)
//! cse7 -> cse0 + cse6
//! ```
//!
//! Note that each expression only appears once: redundant expressions are assigned the same name.
//! We just need to figure out where to insert the `let` definition for each expression.
//!
//! The algorithm proceeds as follow. We insert the `let` for an identifier at the *highest* site
//! where the identifier is used. This is the most "visible" place for the `let` definition. We
//! also, of course, need to generate the dependencies of an identifier before generating its value
//! (e.g., before `cse4`, we need to generate `cse1` in the example above).
//!
//! The bindings generation phase thus first generates a site map which computes the site 
//! of each symbol. We then recursively generate each expression, maintaining a stack of variable
//! bindings: entry `i` in the stack corresponds to the bindings at site `i`. 
//!
//! The site map for the example above would look as follows:
//!
//! ```weld
//! cse0 -> 0 (used in top level function)
//! cse1 -> 0,1 (used only behind true-branch)
//! cse2 -> 0,2 (used only behind false-branch)
//! cse3 -> 0 (used in top level function)
//! cse4 -> 0,1 (used only behind true-branch)
//! cse5 -> 0,1 (used only behind true-branch)
//! cse6 -> 0 (used in top level function)
//! cse7 -> 0 (used in top level function)
//! ```
//!
//! The 0-site variables can be generated at the top of the top function. The 0,1-site variables
//! are generated behind the branch target (the algorithm recursively ensures that dependencies are
//! generated in order).
//!
//! The final output is:
//!
//! ```weld
//! let cse0 = lookup(w, 1);
//! let cse3 = x > 0;
//! let cse6 = if(cse3,
//!     let cse1 = lookup(v, 0);
//!     let cse4 = cse1 + cse1;
//!     cse4 + cse0,
//!     let cse2 = lookup(z, 0);
//!     cse2
//! );
//! let cse7 = cse0 + cse6;
//! cse7
//! ```
//!
//! Calling the inliner will result in a more sane-looking expression tree:
//!
//! ```weld
//! let cse0 = lookup(w, 1);
//! cse0 + if (x > 0,
//!     let cse1 = lookup(v, 0); cse1 + cse1 + cse0,
//!     lookup(z, 0)
//! )
//! ```
//!
//! This is our final CSE'd output.

use ast::*;
use ast::constructors::*;
use ast::ExprKind::*;

use util::SymbolGenerator;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;

/// Implements the CSE transform.
pub fn common_subexpression_elimination(expr: &mut Expr) {
    use super::inliner;
    if let Lambda { .. } = expr.kind {
        expr.uniquify().unwrap();
        Cse::apply(expr);
        // Put this here instead of in the pass because the expression tree looks strange
        // without it, even in tests.
        inliner::inline_let(expr);
    }
}

/// State for the CSE transformation.
#[derive(Debug)]
struct Cse {
    /// Tracks symbol names.
    sym_gen: SymbolGenerator,
    /// Tracks paths.
    counter: i32,
}

/// Specifies whether an expression should be considered for common subexpression elimination.
///
/// We don't apply CSE to primitives such as literals and identifiers, as well as lambdas, since
/// those can't be assigned to names in Weld.
///
/// Expressions which contain a builder type also cannot be CSE'd, since they are special
/// expressions that can produce side effects (e.g., memory allocations that should not be aliased
/// between two values).  In practice, only NewBuilder calls would not be candidates for CSE: since
/// builder operations must be linear, a common subexpression after uniquify involving `merge` or
/// `for` should never appear in a valid Weld program.
trait UseCse {
    fn use_cse(&self) -> bool;
}

impl UseCse for Expr {
    fn use_cse(&self) -> bool {
        if let Let { .. } = self.kind {
            return true;
        }

        if self.ty.contains_builder() {
            return false;
        }

        match self.kind {
            Literal(_) | Ident(_) | Lambda { .. } => false,
            _ => true
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// A site in a program.
///
/// A site represents a path taken by following "scopes".
struct Site {
    site: Vec<i32>,
}

impl Site {
    /// Creates a new empty site.
    fn new() -> Site {
        Site {
            site: vec![],
        }
    }

    /// Pushes a value to the site's path.
    fn push(&mut self, index: i32) {
        self.site.push(index); 
    }
 
    /// Pops the last value from the site's path.
    fn pop(&mut self) {
        self.site.pop();
    }

    /// Returns the depth (length) of this path.
    fn depth(&self) -> usize {
        self.site.len()
    }

    /// Returns whether this site contains `other`.
    ///
    /// A site contains `a` contains another site `b` if `a` is a substring of `b`.
    ///
    /// Returns `true` if the two sites are equal.
    fn contains(&self, other: &Site) -> bool {
        if self.site.len() > other.site.len() {
            return false;
        }
        self.site.iter().zip(other.site.iter()).all(|(a,b)| a == b)
    }
}

/// An unordered list of sites.
struct SiteList {
    sites: Vec<Site>
}

impl SiteList {
    /// Creates a new site list.
    fn new() -> SiteList {
        SiteList {
            sites: vec![],
        }
    }

    /// Returns whether this site or a parent exists in the list.
    fn parent_exists(&self, site: &Site) -> bool {
        self.get_parent_and_index(site).is_some()
    }

    /// Gets the parent of the site and its index, or the site itself if it exists.
    ///
    /// Returns `None` if no ancestor exists in the list.
    fn get_parent_and_index(&self, site: &Site) -> Option<(usize, &Site)> {
        self.sites.iter()
            .enumerate()
            .filter(|(_, s)| s.contains(&site))
            .next()
    }

    /// Deletes the site with the given index from the list.
    fn delete_index(&mut self, site: usize) {
        self.sites.swap_remove(site);
    }

    /// Adds a new site if that site or a parent of it does not exist in the list.
    fn add_site(&mut self, new: Site) {
        // If any of the existing sites do not contain the new one already...
        if !self.sites.iter().any(|s| s.contains(&new)) {
            // keep only the sites that the new site does not contain, and add the new site.
            self.sites.retain(|s| !new.contains(s));
            self.sites.push(new);
        }
    }
}


/// Maps a Symbol to its value.
type Binding = (Symbol, Expr);

/// Maps a symbol to the sites it should be generated in.
type SiteMap = HashMap<Symbol, Vec<Site>>;

impl Cse {
    /// Apply the CSE transformation on the given expression.
    pub fn apply(expr: &mut Expr) {
        let mut cse = Cse {
            sym_gen: SymbolGenerator::from_expression(expr),
            counter: 0,
        };

        // Maps expressions to symbol names.
        let ref mut bindings = HashMap::new();
        let ref mut aliases = HashMap::new();

        cse.remove_common_subexpressions(expr, bindings, aliases);

        // Convert the bindings map from Expr -> Symbol to Symbol -> Expr.
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

        cse.counter = 0;
        let ref mut sites =  cse.build_site_map(expr, bindings);

        // Debug.
        {
            let mut debug = String::new();
            for (k, v) in bindings.iter() {
                debug.push_str(&format!("{} -> {}\n", k, v.pretty_print()));
            }
            trace!("bindings: {}", debug);
            trace!("Expression after assigning cse symbols: {}", expr.pretty_print());
             let mut debug = String::new();
            for (k, v) in sites.iter() {
                debug.push_str(&format!("{} -> {:?}\n", k, v));
            }
            trace!("sites: {}", debug);
        }

        let ref mut generated = HashSet::new();
        let ref mut stack = vec![];

        cse.counter = 0;
        cse.generate_bindings(expr, bindings, generated, &mut Site::new(), stack, sites);

        trace!("After CSE: {}", expr.pretty_print());
    }

    /// Removes common subexpressions and remove bindings.
    ///
    /// This method updates the bindings map to hold expression to identifier information for each
    /// subexpression in the given expression tree. The expressions can be looked up in the
    /// bindings map to find common subexpressions.
    ///
    /// To handle the alias issue (see inline comment), we also return a mapping for a symbol name
    /// with its aliases.
    fn remove_common_subexpressions(&mut self,
                                    expr: &mut Expr,
                                    bindings: &mut HashMap<Expr, Symbol>,
                                    aliases: &mut HashMap<Symbol, Vec<Symbol>>) {

        use self::UseCse;

        trace!("Remove Common Subexpressions called on {}", expr.pretty_print());

        expr.transform_up(&mut |ref mut e| {
            trace!("Processing {}", e.pretty_print());

            if !e.use_cse() {
                return None;
            }

            match e.kind {
                Let { ref name, ref mut value, ref mut body } => {
                    // Let expressions represent "already CSE'd" values, so we can track them.
                    let taken_value = value.take();
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
                    };
                    Some(*body.take())
                }
                _ => {
                    let e = e.take();
                    let ty = e.ty.clone();

                    let name = bindings.entry(e)
                        .or_insert_with(&mut || self.sym_gen.new_symbol("cse"))
                        .clone();

                    // Replace the expression with its CSE name.
                    let replacement = Expr {
                        ty: ty,
                        kind: Ident(name),
                        annotations: Annotations::new(),
                    };
                    Some(replacement)
                }
            }
        });
    }

    /// Builds a site map for each identifier.
    ///
    /// The site map maps a symbol to the *least deep* (i.e., highest) site in an expression tree.
    fn build_site_map(&mut self,
                       expr: &Expr,
                       bindings: &HashMap<Symbol, Expr>) -> HashMap<Symbol, Vec<Site>> {
        let mut sites = HashMap::new();
        self.build_site_map_helper(expr, bindings, &mut sites, &mut Site::new());
        sites
    }

    /// Adds a site into a list of sites.
    ///
    /// The site is de-duplicated with other sites that supercede this one.
    fn add_site(&self, sites: &mut Vec<Site>, new: Site) {
        // If any of the existing sites do not contain the new one already...
        if !sites.iter().any(|s| s.contains(&new)) {
            // keep only the sites that the new site does not contain, and add the new site.
            sites.retain(|s| !new.contains(s));
            sites.push(new);
        }
    }

    fn build_site_map_helper(&mut self,
                           expr: &Expr,
                           bindings: &HashMap<Symbol, Expr>,
                           site_map: &mut HashMap<Symbol, Vec<Site>>,
                           current_site: &mut Site) {

        let mut handled = true;
        match expr.kind {
            If { ref cond, ref on_true, ref on_false } => {
                self.build_site_map_helper(cond, bindings, site_map, current_site);

                // Update current site for branch targets.
                current_site.push(self.counter);
                self.counter += 1; 
                self.build_site_map_helper(on_true, bindings, site_map, current_site);
                current_site.pop();

                current_site.push(self.counter);
                self.counter += 1; 
                self.build_site_map_helper(on_false, bindings, site_map, current_site);
                current_site.pop();
            }
            Lambda { ref body, .. } => {
                current_site.push(self.counter);
                self.counter += 1; 
                self.build_site_map_helper(body, bindings, site_map, current_site);
                current_site.pop();
            }
            Ident(ref name) if bindings.contains_key(name) => {

                // Check if we already generated the identifier definition at a site higher than
                // the one we're currently at.
                let resolved = if site_map.contains_key(name) {
                    let site_list = site_map.get(&name).unwrap();
                    site_list.iter()
                        .enumerate()
                        .filter(|(_, s)| s.contains(current_site)).next().is_some()
                } else {
                    false
                };

                trace!("SCOPE_BUILDER: Saw identifier {} (sites {:?}) at site {:?}",
                &name,
                site_map.get(&name),
                current_site);

                // Need to do this so visit order is consistent! A better way to do the pathing is
                // to compute the path based on the expression...
                if !resolved {
                    {
                        let ent = site_map.entry(name.clone()).or_insert(vec![]);
                        self.add_site(ent, current_site.clone());
                    }
                    let expr = bindings.get(name).unwrap();
                    self.build_site_map_helper(expr, bindings, site_map, current_site);
                }
            }
            _ => {
                handled = false;
            }
        }

        if !handled {
            for child in expr.children() {
                self.build_site_map_helper(child, bindings, site_map, current_site);
            }
        }
    }

    /// Generates bindings in the expression.
    ///
    /// This injects the definition of each symbol into the correct position in the final
    /// expression. Specifically, a symbol is defined in the highest site that symbol is used in.
    ///
    /// The function updates `expr` with the symbol to value mappings in `bindings`. Bindings that
    /// are already inserted are tracked in `generated`. The `stack` tracks sites to generate the
    /// bindings in (where each site holds bindings to be generated at a particular site in the
    /// expression tree), and `sites` maps each symbol to its highest site.
    fn generate_bindings(&mut self,
                         expr: &mut Expr,
                         bindings: &mut HashMap<Symbol, Expr>,
                         generated: &mut HashSet<Symbol>,
                         current_site: &mut Site,
                         stack: &mut Vec<Vec<Binding>>,
                         sites: &mut SiteMap) {

        // NOTE: Because of the way paths are built, this method must traverse Lambdas and If
        // statements in the exact same order as build_site_map_helper!
        let handled = match expr.kind {
            Lambda { ref mut body, .. } => {
                self.generate_bindings_scoped(body, bindings, generated, current_site, stack, sites);
                true
            }
            If { ref mut cond, ref mut on_true, ref mut on_false } => {
                // Generate bindings for the condition, since we won't recurse down into
                // subexpressions.
                self.generate_bindings(cond, bindings, generated, current_site, stack, sites);

                // We want to keep definitions of If/Else statements within the branch target
                // to prevent moving expressions out from behind a branch condition.

                self.generate_bindings_scoped(on_true, bindings, generated, current_site, stack, sites);
                self.generate_bindings_scoped(on_false, bindings, generated, current_site, stack, sites);
                true
            }
            Ident(ref mut sym) if bindings.contains_key(sym) => {

                // Determines whether the expression was already handled at _this_ site.
                // Note that this is the same check used in build_site_map_helper!
                let resolved_at_site = {
                    let site_list = sites.get(&sym).unwrap();
                    site_list.iter()
                        .enumerate()
                        // this is is_none here instead of is_some because the deletion of the site
                        // indicates that we handled it.
                        .filter(|(_, s)| s.contains(&current_site)).next().is_none()
                };

                if !resolved_at_site {
                    // Generate the definition of this identifier and its dependencies.
                    let mut value = bindings.get(sym).cloned().unwrap();
                    self.generate_bindings(&mut value, bindings, generated, current_site, stack, sites);

                    let site_list = sites.get_mut(&sym).unwrap();

                    trace!("GEN_BINDINGS: Saw identifier {} (sites {:?}) at site {:?}",
                    &sym,
                    site_list,
                    current_site);

                    // There will be at most one parent site from the current site where this
                    // identifier should be generated.
                    let delete_index = {
                        let gen_site = site_list.iter().enumerate().filter(|(_, s)| s.contains(&current_site)).next();
                        if let Some(ref result) = gen_site {
                            let gen_site = result.1;
                            generated.insert(sym.clone());
                            let index = gen_site.depth() - 1;
                            let binding_list = stack.get_mut(index).unwrap();
                            binding_list.push((sym.clone(), value));
                            Some(result.0)
                        } else {
                            None
                        }
                    };

                    if let Some(delete_index) = delete_index {
                        site_list.swap_remove(delete_index);
                    }
                }
                true
            }
            _ => false,
        };

        if !handled {
            for child in expr.children_mut() {
                self.generate_bindings(child, bindings, generated, current_site, stack, sites);
            }
        }
    }

    /// Creates a new site (e.g., for a function or if statement) and generates bindings.
    ///
    /// The new site represents a new site in the expression tree to insert let statements.
    ///
    /// The created bindings may not necessarily be in the newest site -- bindings for a symbol
    /// `sym` are generated in the site defined by `sites` (i.e., in the site
    /// `stack[sites.get(sym)]`).
    fn generate_bindings_scoped(&mut self,
                               expr: &mut Expr,
                               bindings: &mut HashMap<Symbol, Expr>,
                               generated: &mut HashSet<Symbol>,
                               current_site: &mut Site,
                               stack: &mut Vec<Vec<(Symbol, Expr)>>,
                               sites: &mut HashMap<Symbol, Vec<Site>>) {

        trace!("Processing sited expression {} (old site={:?}, counter={}",
               expr.pretty_print(),
               current_site,
               self.counter);

        current_site.push(self.counter);
        self.counter += 1; 

        stack.push(vec![]);

        self.generate_bindings(expr, bindings, generated, current_site, stack, sites);

        // Generate the Let expressions for the bindings in this site.
        let binding_list = stack.pop().unwrap();
        let mut prev = expr.take();
        // Reverse list since the tree of let statements is built bottom-up, but evaluated
        // top-down.
        for (sym, expr) in binding_list.into_iter().rev() {
            generated.remove(&sym);
            prev = let_expr(sym, expr, prev).unwrap();
        }

        current_site.pop();

        // Update the expression to contain the Lets.
        *expr = prev;
        trace!("Replaced sited expression with {}", expr.pretty_print());
    }
}

#[cfg(test)]
fn check_cse(input: &str, expect: &str) {
    use tests::check_transform;
    check_transform(input, expect, common_subexpression_elimination);
}

#[test]
fn basic_test() {
    let input = "|| (1+2) + (1+2)";
    let expect = "|| let cse = (1+2); cse + cse";
    check_cse(input, expect);
}

#[test]
fn many_subexprs_test() {
    let input = "|| (1+2) + (1+2) + (3+4) + (3+4)";
    let expect = "|| let cse1 = (1+2); let cse2 = (3+4); cse1 + cse1 + cse2 + cse2";
    check_cse(input, expect);
}

#[test]
fn nesting_test() {
    let input = "|| ((1+2) + (1+2)) + ((1+2) + (1+2))";
    let expect = "|| let cse1 = (1+2); let cse2 = (cse1+cse1); cse2 + cse2";
    check_cse(input, expect);
}

#[test]
fn if_test() {
    let input = "|x: i32, v:vec[i32]| if(x>0, lookup(v,0L), lookup(v,0L))";
    // Currently, CSE does not hoist out of Ifs unless the expression appears outside the If as
    // well.
    let expect = input;
    check_cse(input, expect);
}

#[test]
fn if_test_2() {
    let input = "|x:i32| if(x>0, (1+2) + (1+2), (1+2)+(1+2))";
    let expect = "|x:i32| if(x>0,
        let cse1 = (1+2); cse1 + cse1,
        let cse2 = (1+2); cse2 + cse2)";
    check_cse(input, expect);
}

#[test]
fn if_test_3() {
    let input = "|x:i32| (1+2) + if(x>0, (1+2) + (1+2), (1+2)+(1+2))";
    let expect = "|x:i32| let cse = (1+2);
        cse + if(x>0, cse+cse, cse+cse)";
    check_cse(input, expect);
}

#[test]
fn if_test_4() {
    let input = "|x:i32| if(x>0, (1+2) + (1+2), (1+2)+(1+2)) + (1+2)";
    let expect = "|x:i32| let cse = (1+2);
        if(x>0, cse+cse, cse+cse) + cse";
    check_cse(input, expect);
}

#[test]
fn if_test_5() {
    let input = "|x:i32|
      if(x > 0,
        if (x > 1,
          (1+2),
          (2+3)
        ),
        (1+2) + (1+2)
      )";
    
    // Make sure siblings don't cause values to be hoisted out.
    let expect = "|x:i32| if (x > 0,
        if (x > 1, (1+2), (2+3)),
        let cse = (1+2); cse + cse)";
    check_cse(input, expect);
}

#[test]
fn if_test_6() {
    let input = "|x:i32|
      if(x > 0,
        (1 + 2) + (1+2),
        (2 + 3)
      ) + 
      if(x > 1,
        (1 + 2) + (1+2),
        (2 + 3)
      )";

    let expect = "|x:i32|
      if(x > 0,
        let cse = (1+2); cse + cse,
        (2+3)
      ) +
      if(x > 1,
        let cse = (1+2); cse + cse,
        (2+3)
      )";
    check_cse(input, expect);
}

#[test]
fn if_test_7() {
    // Tests cases where we can arrive at a common subexpression via multiple paths.
    //
    // This is important to test because it ensures that the scope map builder and bindings
    // generator traverse expressions in a consistent way.
    //
    // In the input below, we can arrive at (1+2) via two different sites:
    //
    // 0 -> 1, (root, true condition), and
    // 0 -> 3 -> 4 (root, true condition of second if, and true condition of first if).
    //
    // If the site builder and bindings generator handle this differently (e.g., by pruning paths),
    // the count specifying the path will become inconsistent and yield an error, because later
    // expressions will have inconsistent paths.
    let input = "|x:i32|
        let c = (if (x>0, (1+2), 0));
        if (c > 0,
            c + 5,
            (if (x>1, (2+3), 0)) 
        )";
    let expect = input;
    check_cse(input, expect);
}

#[test]
fn for_test() {
    let input = "|| result(for([(1+2)], merger[i32,+], |b,i,e| merge(b, e + (1+2))))";
    let expect = "|| let cse = (1+2); result(for([cse], merger[i32,+], |b,i,e| merge(b, e + cse)))";
    check_cse(input, expect);
}

#[test]
fn for_test_2() {
    let input = "|| result(for([1], merger[i32,+], |b,i,e| merge(b, e + (1+2)))) + (1+2)";
    let expect = "|| let cse = (1+2); result(for([1], merger[i32,+], |b,i,e| merge(b, e + cse))) + cse";
    check_cse(input, expect);
}

#[test]
fn for_test_3() {
    let input = "|| result(for([1], merger[i32,+], |b,i,e| merge(b, e + (1+2) + (1+2))))";
    let expect = "|| result(for([1], merger[i32,+], |b,i,e| let cse = (1+2); merge(b, e + cse + cse)))";
    check_cse(input, expect);
}


#[test]
fn builder_test() {
    let input = "|| {appender[i32], appender[i32]}";
    let expect = input;
    check_cse(input, expect);
}

#[test]
fn builder_test_2() {
    let input = "|| {result(appender[i32]), result(appender[i32])}";
    let expect = "|| let cse = result(appender[i32]); {cse, cse}";
    check_cse(input, expect);
}


#[test]
fn builder_test_3() {
    let input = "|| {{appender[i32], appender[i32], 1}, {appender[i32], appender[i32], 1}}";
    let expect = input;
    check_cse(input, expect);
}

#[test]
fn builder_test_4() {
    let input = "|| {{appender[i32], appender[i32], (1+2)}, {appender[i32], appender[i32], (1+2)}}";
    let expect = "|| let cse = (1+2); {{appender[i32], appender[i32], cse}, {appender[i32], appender[i32], cse}}";
    check_cse(input, expect);
}

#[test]
fn alias_test() {
    let input = "|x:i32| let a = x; let b = x; a + a + b + b";
    let expect = input;
    check_cse(input, expect);
}


#[test]
fn let_test() {
    let input = "|x:i32| let a = (1+2); (1+2) + a + (1+2)";
    let expect = "|x:i32| let cse = (1+2); cse + cse + cse";
    check_cse(input, expect);
}
