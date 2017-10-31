use super::ast::*;
use super::error::*;

use super::transforms::loop_fusion;
use super::transforms::inliner;
use super::transforms::size_inference;
use super::transforms::annotator;
use super::transforms::vectorizer;

use super::expr_hash::*;

use std::collections::HashMap;

pub struct Pass {
    transforms: Vec<fn(&mut Expr<Type>)>,
    pass_name: String,
}

/// Manually implement Clone for Pass because it cannot be #derived due to the fn type inside it.
impl Clone for Pass {
    fn clone(&self) -> Pass {
        Pass {
            transforms: self.transforms.iter().map(|p| *p).collect::<Vec<_>>(),
            pass_name: self.pass_name.clone()
        }
    }
}

impl Pass {
    pub fn new(transforms: Vec<fn(&mut Expr<Type>)>, pass_name: &'static str) -> Pass {
        return Pass {
                   transforms: transforms,
                   pass_name: String::from(pass_name),
               };
    }

    pub fn transform(&self, mut expr: &mut Expr<Type>) -> WeldResult<()> {
        let mut continue_pass = true;
        let mut before = ExprHash::from(expr)?.value();
        while continue_pass {
            for transform in &self.transforms {
                transform(&mut expr);
            }
            let after = ExprHash::from(expr)?.value();
            continue_pass = !(before == after);
            before = after;
        }
        Ok(())
    }

    pub fn pass_name(&self) -> String {
        self.pass_name.clone()
    }
}

lazy_static! {
    pub static ref OPTIMIZATION_PASSES: HashMap<&'static str, Pass> = {
        let mut m = HashMap::new();
        m.insert("inline-apply",
                 Pass::new(vec![inliner::inline_apply], "inline-apply"));
        m.insert("inline-let",
                 Pass::new(vec![inliner::inline_let], "inline-let"));
        m.insert("inline-zip",
                 Pass::new(vec![inliner::inline_zips], "inline-zip"));
        m.insert("loop-fusion",
                 Pass::new(vec![loop_fusion::fuse_loops_horizontal,
                                loop_fusion::fuse_loops_vertical,
                                inliner::inline_get_field],
                 "loop-fusion"));
        m.insert("infer-size",
                 Pass::new(vec![size_inference::infer_size],
                 "infer-size"));
        m.insert("predicate",
                 Pass::new(vec![vectorizer::predicate],
                 "predicate"));
        m.insert("vectorize",
                 Pass::new(vec![vectorizer::vectorize],
                 "vectorize"));
        m.insert("fix-iterate",
                 Pass::new(vec![annotator::force_iterate_parallel_fors],
                 "fix-iterate"));

        m
    };
}
