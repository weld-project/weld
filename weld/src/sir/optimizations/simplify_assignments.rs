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

use crate::sir::StatementKind::{Assign, AssignLiteral};
use crate::sir::*;

/// Applies the simplifying assignments pass.
pub fn simplify_assignments(prog: &mut SirProgram) -> WeldResult<()> {
    for func in prog.funcs.iter_mut() {
        simplify_assignments_in_function(func)?
    }
    Ok(())
}

fn simplify_assignments_in_function(func: &mut SirFunction) -> WeldResult<()> {
    // XXX This is a hack! Currently, the code gen assumes that the symbol names in a for loop body
    // will match the symbol names for the same data in the calling function (e.g., if the vector
    // to loop over is called `data1` in the calling function, it must be called `data1` in the
    // loop body). If this pass deletes `data1` because of a redundant assignment, it will cause
    // an error when compiling the loop body function unless this pass becomes interprocedural. It
    // will be better to just simplify the SIR so that the loop body functions are self contained
    // and don't make any assumptions about symbol names -- they should just refer to their own
    // parameters instead.
    //
    // For now, we circumvent deleting these symbols by just applying this optimization to
    // functions that are loop bodies and are innermost loops (i.e., are guaranteed to not call
    // another for loop).
    if !func.loop_body || !func.innermost_loop {
        return Ok(());
    }

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
                    Some(ref current_source) => *current_source != source,
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
                // Remove the target in case it was added before.
                assignments.remove(statement.output.as_ref().unwrap());
            }
        }
    }

    // Delete assignments that are still in the assigments map.
    for block in func.blocks.iter_mut() {
        block.statements.retain(|ref statement| {
            if let Assign(_) = statement.kind {
                let target = statement.output.as_ref().unwrap();
                !assignments.contains_key(target)
            } else {
                true
            }
        });

        // Perform the substitution.
        for (key, value) in assignments.iter() {
            block.substitute_symbol(key, value);
        }
    }

    Ok(())
}
