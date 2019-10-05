//! An SIR pass that removes unnecessary assignments.
//!
//! This pass replaces assignment expressions that assign the same source to a particular target in
//! each basic block with the source directly. For example:
//!
//! ```sir
//! B1:
//!    fn1_tmp__0 = x
//!    jump B3
//! B2:
//!    fn1_tmp__0 = x
//!    jump B3
//! B3:
//!    return  fn1_tmp__0
//! ```
//!
//! will become:
//!
//! ```sir
//! B1:
//!    jump B3
//! B2:
//!    jump B3
//! B3:
//!    return x
//! ```

use fnv;

use crate::sir::*;
use crate::sir::StatementKind::{Assign, AssignLiteral};

/// Applies the simplifying assignments pass.
pub fn simplify_assignments(prog: &mut SirProgram) -> WeldResult<()> {
    for func in prog.funcs.iter_mut() {
        simplify_assignments_in_function(func)?
    }
    Ok(())
}

fn simplify_assignments_in_function(func: &mut SirFunction) -> WeldResult<()> {
    // Assignments that are redundant and should be deleted.
    let mut assignments = fnv::FnvHashMap::default();
    // Valid assignments where the target variable is assigned different source variables in
    // different basic blocks.
    let mut validset = fnv::FnvHashSet::default();

    // Find the assignments that should be simplified by checking whether each assignment is
    // assigned different values in different basic blocks.
    for block in func.blocks.iter() {
        for statement in block.statements.iter() {
            if let Assign(ref source) = statement.kind {
                let target = statement.output.as_ref().unwrap();

                // We've already seen this target and shown that it is valid.
                if validset.contains(target) {
                    continue;
                }

                // The assignment is valid if different basic blocks assign the target different
                // values.
                let is_valid = match assignments.get(target) {
                    Some(ref current_source) => {
                        *current_source != source
                    }
                    None => {
                        // We haven't seen an assigment to this target before.
                        assignments.insert(target.clone(), source.clone());
                        false
                    }
                };

                // Move to validset if the assignment is valid.
                if is_valid {
                    validset.insert(assignments.remove(target).unwrap());
                }
            }
            
            // Always treat assignments to a literal as valid.
            if let AssignLiteral(_) = statement.kind {
                validset.insert(statement.output.clone().unwrap());
            }
        }
    }

    // Delete assignments that are still in the assigments map.
    for block in func.blocks.iter_mut() {
        block.statements.retain(|ref statement| {
            if let Assign(_) = statement.kind {
                !assignments.contains_key(statement.output.as_ref().unwrap())
            } else {
                true
            }
        });

        // Perform the substitution.
        for (key, value) in assignments.iter() {
            block.substitute_symbol(key, value)
        }
    }

    Ok(())
}
