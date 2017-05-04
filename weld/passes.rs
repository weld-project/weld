use super::ast::*;
use super::error::*;
use super::transforms;


pub struct Pass {
    transform_fns: Vec<fn(&mut Expr<Type>)>,
    transform_names: Vec<String>,
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

pub fn get_pass(pass_name: String) -> WeldResult<Pass> {
    match pass_name.as_ref() {
        "inline-apply" =>
            Ok(Pass {
                transform_fns: vec![transforms::inline_apply],
                transform_names: vec![String::from("Inline apply")],
                pass_name: String::from("inline-apply")
            }),
        "inline-let" =>
            Ok(Pass {
                transform_fns: vec![transforms::inline_let],
                transform_names: vec![String::from("Inline let")],
                pass_name: String::from("inline-let")
            }),
        "inline-zip" =>
            Ok(Pass {
                transform_fns: vec![transforms::inline_zips],
                transform_names: vec![String::from("Inline zip")],
                pass_name: String::from("inline-zip")
            }),
        "loop-fusion" =>
            Ok(Pass {
                transform_fns: vec![transforms::fuse_loops_horizontal,
                                    transforms::fuse_loops_vertical],
                transform_names: vec![String::from("Horizontal loop fusion"),
                                    String::from("Vertical loop fusion")],
                pass_name: String::from("loop-fusion")
            }),
        "uniquify" =>
            Ok(Pass {
                transform_fns: vec![transforms::uniquify],
                transform_names: vec![String::from("Uniquify")],
                pass_name: String::from("uniquify")
            }),
        _ => weld_err!("Undefined pass: {}", pass_name)
    }
}