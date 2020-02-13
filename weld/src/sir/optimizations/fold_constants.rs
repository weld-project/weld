//! Folds constants in SIR programs.
//!
//! This transform walks each SIR function and attempts to evaluate expressions at runtime,
//! deleting definitions and variables which evaluate to simple constants.
//!
//! The transform in conservative in two ways:
//!
//! 1. It will not remove symbols required by terminators, in case a code generator relies on the
//! existence of these symbols.
//!
//! 2. It will not remove symbols which are used as parameters to a function (though this can
//!    probably be changed to support a simple inteprocedural constant simplification).

use std::collections::BTreeMap;

use crate::error::*;
use crate::sir::*;

use fnv;

/// Evaluates a binary operation over two literals, returning a new literal.
// TODO(shoumik): Maybe we should just implement this as a method or as traits (e.g., Add, Sub,
// etc.) on LiteralKind? Not every literal implements every binop, however...
fn evaluate_binop(
    kind: BinOpKind,
    left: LiteralKind,
    right: LiteralKind,
) -> WeldResult<LiteralKind> {
    use crate::ast::BinOpKind::*;
    use crate::ast::LiteralKind::*;
    let result = match kind {
        // Just support the basics for now.
        Add => match (left, right) {
            (I8Literal(l), I8Literal(r)) => I8Literal(l + r),
            (I16Literal(l), I16Literal(r)) => I16Literal(l + r),
            (I32Literal(l), I32Literal(r)) => I32Literal(l + r),
            (I64Literal(l), I64Literal(r)) => I64Literal(l + r),
            (F32Literal(l), F32Literal(r)) => {
                F32Literal((f32::from_bits(l) + f32::from_bits(r)).to_bits())
            }
            (F64Literal(l), F64Literal(r)) => {
                F64Literal((f64::from_bits(l) + f64::from_bits(r)).to_bits())
            }
            _ => {
                return compile_err!("Mismatched types in evaluate_binop");
            }
        },
        Subtract => match (left, right) {
            (I8Literal(l), I8Literal(r)) => I8Literal(l - r),
            (I16Literal(l), I16Literal(r)) => I16Literal(l - r),
            (I32Literal(l), I32Literal(r)) => I32Literal(l - r),
            (I64Literal(l), I64Literal(r)) => I64Literal(l - r),
            (F32Literal(l), F32Literal(r)) => {
                F32Literal((f32::from_bits(l) - f32::from_bits(r)).to_bits())
            }
            (F64Literal(l), F64Literal(r)) => {
                F64Literal((f64::from_bits(l) - f64::from_bits(r)).to_bits())
            }
            _ => {
                return compile_err!("Mismatched types in evaluate_binop");
            }
        },
        Multiply => match (left, right) {
            (I8Literal(l), I8Literal(r)) => I8Literal(l * r),
            (I16Literal(l), I16Literal(r)) => I16Literal(l * r),
            (I32Literal(l), I32Literal(r)) => I32Literal(l * r),
            (I64Literal(l), I64Literal(r)) => I64Literal(l * r),
            (F32Literal(l), F32Literal(r)) => {
                F32Literal((f32::from_bits(l) * f32::from_bits(r)).to_bits())
            }
            (F64Literal(l), F64Literal(r)) => {
                F64Literal((f64::from_bits(l) * f64::from_bits(r)).to_bits())
            }
            _ => {
                return compile_err!("Mismatched types in evaluate_binop");
            }
        },
        Divide => match (left, right) {
            (I8Literal(l), I8Literal(r)) => I8Literal(l / r),
            (I16Literal(l), I16Literal(r)) => I16Literal(l / r),
            (I32Literal(l), I32Literal(r)) => I32Literal(l / r),
            (I64Literal(l), I64Literal(r)) => I64Literal(l / r),
            (F32Literal(l), F32Literal(r)) => {
                F32Literal((f32::from_bits(l) / f32::from_bits(r)).to_bits())
            }
            (F64Literal(l), F64Literal(r)) => {
                F64Literal((f64::from_bits(l) / f64::from_bits(r)).to_bits())
            }
            _ => {
                return compile_err!("Mismatched types in evaluate_binop");
            }
        },
        _ => {
            return compile_err!("Unsupported binary operation in evaluate_binop");
        }
    };
    Ok(result)
}

pub fn fold_constants(prog: &mut SirProgram) -> WeldResult<()> {
    let parameters = &mut fnv::FnvHashSet::default();
    // Collect all the Symbols passed between functions. We will keep the
    // definitions of these symbols intact (even if we assign simple literal
    // values to them).
    //
    // TODO(shoumik): We could also run an interprocedural version of this transform
    // that eliminates passing these, instead propagating the computed value through
    // the program.
    for func in prog.funcs.iter() {
        parameters.extend(func.params.iter().map(&|(k, _)| k).cloned());
        // Add these because code gen expects them to be present even if they are unused.
        for block in func.blocks.iter() {
            parameters.extend(block.terminator.children().cloned());
        }
    }

    for func in prog.funcs.iter_mut() {
        fold_constants_in_function(func, parameters)?;
    }
    Ok(())
}

fn fold_constants_in_function(
    func: &mut SirFunction,
    global_params: &fnv::FnvHashSet<Symbol>,
) -> WeldResult<()> {
    use crate::sir::StatementKind::*;
    let mut assignment_counts: fnv::FnvHashMap<Symbol, i32> = fnv::FnvHashMap::default();
    for block in func.blocks.iter_mut() {
        for statement in block.statements.iter_mut() {
            if statement.output.is_some() {
                let assignment_count = assignment_counts
                    .entry(statement.output.clone().unwrap())
                    .or_insert(0);
                *assignment_count += 1;
            }
        }
    }

    // Maps a Symbol to a known literal value.
    let mut values: fnv::FnvHashMap<Symbol, LiteralKind> = fnv::FnvHashMap::default();
    // Set of used symbols in a function.
    let mut used_symbols = fnv::FnvHashSet::default();

    // Initialize the used_symbols with all builder-typed expressions. These expressions may be
    // "unused" in the SIR, but backends may represent assignments of them via mutable state, so
    // removing them is potentially unsafe.
    for block in func.blocks.iter() {
        for statement in block.statements.iter() {
            if let Some(ref sym) = statement.output {
                if func.symbol_type(sym)?.is_builder() {
                    used_symbols.insert(sym.clone());
                }
            }
        }
    }

    // Mark the loop variables as used - again, even if they are technically unused,
    // backends may generate them unconditionally.
    for var in func.loop_variables.iter() {
        used_symbols.insert(var.clone());
    }

    for block in func.blocks.iter_mut() {
        for statement in block.statements.iter_mut() {
            let replacement_lit = match statement.kind {
                // Literal value - add it to the map
                AssignLiteral(ref lit) => {
                    let output_sym = statement.output.clone().unwrap();
                    if assignment_counts[&output_sym] == 1 {
                        values.insert(output_sym, (*lit).clone());
                        Some((*lit).clone())
                    } else {
                        None
                    }
                }
                // Aliases
                Assign(ref sym) => {
                    // contains_key instead of get to avoid double mutable borrow
                    if values.contains_key(sym) {
                        let output_sym = statement.output.clone().unwrap();
                        let value = values[sym].clone();
                        if assignment_counts[&output_sym] == 1 {
                            values.insert(output_sym, value.clone());
                        }
                        Some(value)
                    } else {
                        None
                    }
                }
                BinOp {
                    ref op,
                    ref left,
                    ref right,
                } if (&values).contains_key(left) && (&values).contains_key(right) => {
                    let left_val = values[left].clone();
                    let right_val = values[right].clone();
                    // If this throws an error, it just means that we don't support evaluating the
                    // expression right now.
                    if let Ok(result) = evaluate_binop(*op, left_val, right_val) {
                        let output_sym = statement.output.clone().unwrap();
                        if assignment_counts[&output_sym] == 1 {
                            values.insert(output_sym, result.clone());
                        }
                        Some(result)
                    } else {
                        None
                    }
                }
                // Unsupported for now
                _ => None,
            };

            if let Some(val) = replacement_lit {
                // If this is true, we computed a constant value for this statement. Replace it
                // with that constant value.
                let kind = AssignLiteral(val);
                let new_statement = Statement::new(statement.output.clone(), kind);
                *statement = new_statement;
            } else {
                // We didn't replace this statement with an AssignLiteral, so record the symbols
                // which it uses.
                used_symbols.extend(statement.kind.children().cloned());
            }
        }
    }

    // Delete unused symbols. We'll only delete the locals, since deleting the parameters will
    // require modifying global state.
    for block in func.blocks.iter_mut() {
        block.statements.retain(|ref s| {
            // Keep the statement that defines the symbol if it is used on the RHS of some
            // expression, and remove it otherwise.
            if let Some(ref sym) = s.output {
                used_symbols.contains(sym) || global_params.contains(sym)
            } else {
                // If there's no output symbol (i.e., it's a `Merge`), retain it.
                true
            }
        });
    }

    // TODO(shoumik): is there a better way of doing this...
    let mut locals = BTreeMap::new();
    for (k, v) in func.locals.iter() {
        if used_symbols.contains(k) || global_params.contains(k) {
            locals.insert(k.clone(), v.clone());
        }
    }
    func.locals = locals;
    Ok(())
}
