use super::ast::*;
use super::error::*;
use super::transforms;

use std::collections::HashMap;


pub struct Pass {
    transform_fns: Vec<fn(&mut Expr<Type>)>,
    pass_name: String
}

impl Pass {
    pub fn transform(&self, mut expr: &mut Expr<Type>) -> WeldResult<()> {
        let mut expr_copy = expr.clone();
        let mut continue_pass = true;
        while continue_pass {
            for transform_fn in &self.transform_fns {
                transform_fn(&mut expr);
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
        m.insert("inline-apply", Pass {
            transform_fns: vec![transforms::inline_apply],
            pass_name: String::from("inline-apply")
        });
        m.insert("inline-let", Pass {
            transform_fns: vec![transforms::inline_let],
            pass_name: String::from("inline-let")
        });
        m.insert("inline-zip", Pass {
            transform_fns: vec![transforms::inline_zips],
            pass_name: String::from("inline-zip")
        });
        m.insert("loop-fusion", Pass {
            transform_fns: vec![transforms::fuse_loops_horizontal,
                                transforms::fuse_loops_vertical],
            pass_name: String::from("loop-fusion")
        });
        m.insert("uniquify", Pass {
            transform_fns: vec![transforms::uniquify],
            pass_name: String::from("uniquify")
        });
        m
    };
}