//! Optimizer over the Weld AST.
//!
//! This module manages optimizations over the Weld AST. Optimizations are represented as
//! transformations in the `transforms` module, which convert one AST to another using rule-based
//! pattern matching. The module provides a pass interface that groups related transforms into a
//! pass, and also provides utilities for applying passes until a fix point (that is, until the
//! pass stops modifying the AST).

use time;

use time::PreciseTime;

use crate::ast::*;
use crate::error::*;
use crate::util::stats::CompilationStats;

pub use self::passes::*;

mod passes;
pub mod transforms;

/// Apply passes from a list until fix point.
pub fn apply_passes(
    expr: &mut Expr,
    passes: &[Pass],
    stats: &mut CompilationStats,
    use_experimental: bool,
) -> WeldResult<()> {
    for pass in passes {
        let start = PreciseTime::now();
        pass.transform(expr, use_experimental)?;
        let end = PreciseTime::now();
        stats.pass_times.push((pass.pass_name(), start.to(end)));
        debug!("After {} pass:\n{}", pass.pass_name(), expr.pretty_print());
    }
    Ok(())
}
