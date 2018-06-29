//! Optimizer over the Weld AST.
//!
//! This module manages optimizations over the Weld AST. Optimizations are represented as
//! transformations in the `transforms` module, which convert one AST to another using rule-based
//! pattern matching. The module provides a pass interface that groups related transforms into a
//! pass, and also provides utilities for applying passes until a fix point (that is, until the
//! pass stops modifying the AST).

extern crate time;

use time::PreciseTime;

use ast::*;
use error::*;
use util::stats::CompilationStats;

pub use self::passes::*;

pub mod transforms;
mod passes;

/// Apply passes from a list until fix point.
pub fn apply_passes(expr: &mut Expr,
                        passes: &Vec<Pass>,
                        stats: &mut CompilationStats,
                        use_experimental: bool) -> WeldResult<()> {
    for pass in passes {

        if pass.pass_name() == "vectorize" {
            continue;
        }
        
        let start = PreciseTime::now();
        pass.transform(expr, use_experimental)?;
        let end = PreciseTime::now();
        stats.pass_times.push((pass.pass_name(), start.to(end)));
        debug!("After {} pass:\n{}", pass.pass_name(), expr.pretty_print());
    }
    Ok(())
}
