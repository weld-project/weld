//! Evaluate Weld ASTs at compile time (by interpreting them).

/*

use std::any::Any;
use std::collections::HashMap;

use super::ast::*;
use super::ast::Expr::*;
use super::ast::Type::*;
use super::ast::ScalarKind::*;
use super::ast::BinOpKind::*;
use super::error::*;

/// Evaluate a subset of Weld expressions at compile time (by interpreting them).
pub fn evaluate(expr: &Expr) -> WeldResult<Box<Any>> {
    let mut env: HashMap<Symbol, Binding> = HashMap::new();
    evaluate_with_env(&mut env, expr)
}

struct Binding(Type, Box<Any>);

fn evaluate_with_env(env: &mut HashMap<Symbol, Binding>, expr: &Expr) -> WeldResult<Box<Any>> {
    match *expr {
        I32Literal(value) => Ok(Box::new(value)),

        BinOp(Scalar(I32), kind, ref left, ref right) => {
            let left: i32 = *(try!(evaluate_with_env(env, left)).downcast::<i32>().unwrap());
            let right: i32 = *(try!(evaluate_with_env(env, right)).downcast::<i32>().unwrap());
            let res: WeldResult<i32> = match kind {
                Add => Ok(left + right),
                Subtract => Ok(left - right),
                Multiply => Ok(left * right),
                Divide => if right != 0 { Ok(left / right) } else { weld_err!("division by 0") }
            };
            res.map(|r| Box::new(r) as Box<Any>)
        },

        Ident(ref itype, ref symbol) => match env.get(symbol) {
            Some(&Binding(ref btype, ref value)) => {
                if *btype != *itype {
                    return Err(WeldError(format!(
                        "Wrong binding for {:?}: {:?} instead of {:?}", symbol, btype, itype)))
                }
                match *itype {
                    Scalar(I32) => Ok(Box::new(*value.downcast_ref::<i32>().unwrap())),
                    _ => weld_err!("Unsupported expression")
                }
            },
            None => Err(WeldError(format!("Binding not found: {:?}: {:?}", symbol, itype)))
        },

        Let { ref symbol, ref value, ref body, .. } => {
            let value_res = try!(evaluate_with_env(env, value));
            let old_entry = env.insert(symbol.clone(), Binding(get_type(&value), value_res));
            let result = evaluate_with_env(env, body);
            match old_entry {
                Some(old) => env.insert(symbol.clone(), old),
                None => env.remove(symbol)
            };
            result
        },

        _ => weld_err!("Unsupported expression"),
    }
}

*/