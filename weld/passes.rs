use super::ast::*;
use super::error::*;
use super::transforms;
use super::vectorizer;

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
        let mut expr_copy = expr.clone();
        let mut continue_pass = true;
        while continue_pass {
            for transform in &self.transforms {
                transform(&mut expr);
            }

            continue_pass = !try!(expr.compare_ignoring_symbols(&expr_copy));
            expr_copy = expr.clone();
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
                 Pass::new(vec![transforms::inline_apply], "inline-apply"));
        m.insert("inline-let",
                 Pass::new(vec![transforms::inline_let], "inline-let"));
        m.insert("inline-zip",
                 Pass::new(vec![transforms::inline_zips], "inline-zip"));
        m.insert("loop-fusion",
                 Pass::new(vec![transforms::fuse_loops_horizontal,
                                transforms::fuse_loops_vertical,
                                transforms::simplify_get_field],
                 "loop-fusion"));
        m.insert("vectorize",
                 Pass::new(vec![vectorizer::vectorize],
                 "vectorize"));
        m
    };
}
