//! Common transformations on expressions.

use super::ast::*;
use super::ast::ExprKind::*;
use super::ast::Type::*;
use super::ast::BuilderKind::*;
use super::ast::LiteralKind::*;
use super::error::*;

use std::collections::HashMap;

use super::util::SymbolGenerator;

/// Inlines Zip expressions as collections of iters. Using Zips outside of a For loop is currently
/// unsupported behavior. This transform handles the simple case of converting Zips in macros
/// such as map and filter into Iters in For loops.
///
/// TODO(shoumik): Perhaps Zip should just be a macro? Then macros need to be ordered.
pub fn inline_zips(expr: &mut Expr<Type>) {
    expr.transform(&mut |ref mut e| {
        if let For {
                   ref mut iters,
                   ref builder,
                   ref func,
               } = e.kind {
            if iters.len() == 1 {
                let ref first_iter = iters[0];
                if let Zip { ref vectors } = first_iter.data.kind {
                    let new_iters = vectors
                        .iter()
                        .map(|v| {
                                 Iter {
                                     data: Box::new(v.clone()),
                                     start: None,
                                     end: None,
                                     stride: None,
                                 }
                             })
                        .collect::<Vec<_>>();
                    return Some(Expr {
                                    ty: e.ty.clone(),
                                    kind: For {
                                        iters: new_iters,
                                        builder: builder.clone(),
                                        func: func.clone(),
                                    },
                                    annotations: Annotations::new(),
                                });
                }
            }
        }
        None
    });
}

/// Modifies symbol names so each symbol is unique in the AST. This transform should be applied
/// "up front" and downstream transforms shoud then use SymbolGenerator to generate new unique
/// symbols.
///
/// Returns an error if an undeclared symbol appears in the program.
pub fn uniquify<T: TypeBounds>(expr: &mut Expr<T>) -> WeldResult<()> {
    // Maps a string name to its current integer ID in the current scope.
    let mut id_map: HashMap<Symbol, i32> = HashMap::new();
    // Maps a symbol name to the the maximum ID observed for it.
    let mut max_ids: HashMap<String, i32> = HashMap::new();
    _uniquify(expr, &mut id_map, &mut max_ids)
}

/// Helper function for uniquify.
fn _uniquify<T: TypeBounds>(expr: &mut Expr<T>,
                            id_map: &mut HashMap<Symbol, i32>,
                            max_ids: &mut HashMap<String, i32>)
                            -> WeldResult<()> {

    // Given a newly defined Symbol sym, increments the symbol's ID and returns a new symbol.
    let push_id =
        |id_map: &mut HashMap<Symbol, i32>, max_ids: &mut HashMap<String, i32>, sym: &Symbol| {
            let max_id = max_ids.entry(sym.name.clone()).or_insert(-1);
            let id = id_map.entry(sym.clone()).or_insert(*max_id);
            *id += 1;
            if *id > *max_id {
                *max_id = *id;
            }
            Symbol::new(&sym.name.clone(), *id)
        };

    // Decrements the ID for a given symbol name and returns the new symbol If the symbol is not
    // found, it was undefined in the current expression.
    let pop_id = |id_map: &mut HashMap<Symbol, i32>, sym: &Symbol| {
        let id = id_map.entry(sym.clone()).or_insert(-1);
        *id -= 1;
        Symbol::new(&sym.name.clone(), *id)
    };

    // Returns the current ID for a given defined symbol. If the symbol is not found, it was
    // undefined in the expression.
    let get_id = |id_map: &mut HashMap<Symbol, i32>, sym: &Symbol| {
        let id = match id_map.get(sym) {
            Some(e) => e,
            _ => {
                return weld_err!("Undefined symbol {} in uniquify", sym.name);
            }
        };
        Ok(Symbol::new(&sym.name.clone(), *id))
    };

    let mut retval = Ok(());

    // Walk the expression tree, maintaining a map of the highest ID seen for a given
    // symbol. The ID is incremented when a symbol is redefined in a new scope, and
    // decremented when exiting the scope. Symbols seen in the original expression
    // are tracked as keys.
    expr.transform_and_continue(&mut |ref mut e| {
        if let Ident(ref sym) = e.kind {
            let gid = match get_id(id_map, sym) {
                Ok(e) => e,
                Err(err) => {
                    retval = Err(err);
                    return (None, false);
                }
            };
            return (Some(Expr {
                             ty: e.ty.clone(),
                             kind: Ident(gid),
                             annotations: Annotations::new(),
                         }),
                    false);
        } else if let Lambda {
                          ref mut params,
                          ref mut body,
                      } = e.kind {
            // Create new parameters for the lambda that will replace this one.
            let new_params = params
                .iter()
                .map(|ref p| {
                         Parameter {
                             ty: p.ty.clone(),
                             name: push_id(id_map, max_ids, &p.name),
                         }
                     })
                .collect::<Vec<_>>();

            if let Err(err) = _uniquify(body, id_map, max_ids) {
                retval = Err(err);
                return (None, false);
            }
            for param in params.iter() {
                pop_id(id_map, &param.name);
            }

            return (Some(Expr {
                             ty: e.ty.clone(),
                             kind: Lambda {
                                 params: new_params,
                                 body: body.clone(),
                             },
                             annotations: Annotations::new(),
                         }),
                    false);
        } else if let Let {
                          ref mut name,
                          ref mut value,
                          ref mut body,
                      } = e.kind {
            if let Err(err) = _uniquify(value, id_map, max_ids) {
                retval = Err(err);
                return (None, false);
            }
            let new_sym = push_id(id_map, max_ids, &name);
            if let Err(err) = _uniquify(body, id_map, max_ids) {
                retval = Err(err);
                return (None, false);
            }
            return (Some(Expr {
                             ty: e.ty.clone(),
                             kind: Let {
                                 name: new_sym,
                                 value: value.clone(),
                                 body: body.clone(),
                             },
                             annotations: Annotations::new(),
                         }),
                    false);
        }
        (None, true)
    });
    retval
}

/// Inlines Apply nodes whose argument is a Lambda expression. These often arise during macro
/// expansion but it's simpler to inline them before doing type inference.
/// Unlike many of the other transformations, we make this one independent of types so that
/// we can apply it before type inference.
///
/// Caveats:
/// - Functions that reuse a parameter twice have its expansion appear twice, instead of assigning
///   it to a temporary as would happen with function application.
/// - Does not complete inlining if some of the functions take functions as arguments (in that
///   case, the expressions after inlining may lead to more inlining).
pub fn inline_apply<T: TypeBounds>(expr: &mut Expr<T>) {
    expr.transform(&mut |ref mut expr| {
        if let Apply {
                   ref func,
                   params: ref args,
               } = expr.kind {
            if let Lambda {
                       ref params,
                       ref body,
                   } = func.kind {
                let mut new = *body.clone();
                for (param, arg) in params.iter().zip(args) {
                    new.substitute(&param.name, &arg);
                }
                return Some(new);
            }
        }
        None
    });
}

/// Inlines Let calls if the symbol defined by the Let statement is used
/// less than one time.
pub fn inline_let(expr: &mut Expr<Type>) {
    expr.transform(&mut |ref mut expr| {
        if let Let {
                   ref mut name,
                   ref mut value,
                   ref mut body,
               } = expr.kind {
            if symbol_usage_count(name, body) <= 1 {
                body.transform(&mut |ref mut expr| {
                    // TODO(shoumik): What about symbol redefinitions?
                    if let Ident(ref symbol) = expr.kind {
                        if symbol == name {
                            return Some(*value.clone());
                        }
                    }
                    return None;
                });
                return Some(*body.clone());
            }
        }
        return None;
    });
}

fn symbol_usage_count(sym: &Symbol, expr: &Expr<Type>) -> u32 {
    let mut usage_count = 0;
    expr.traverse(&mut |ref e| {
        if let For { ref func, .. } = e.kind {
            // The other child expressions of the For will be counted by traverse.
            if symbol_usage_count(sym, func) >= 1 {
                usage_count += 3;
            }
        } else if let Ident(ref symbol) = e.kind {
            if sym == symbol {
                usage_count += 1;
            }
        }
    });

    usage_count
}

/// Fuses for loops over the same vector in a zip into a single for loop which produces a vector of
/// structs directly.
///
/// Some examples:
///
/// for(zip(
///     result(for(a, appender, ...))
///     result(for(a, appender, ...))
/// ), ...)
///
/// will become for(result(for(a, ...))) where the nested for will produce a vector of structs with
/// two elements.
///
/// Caveats:
///     - Like all Zip-based transforms, this function currently assumes that the output of each
///     expression in the Zip is the same length.
///
pub fn fuse_loops_horizontal(expr: &mut Expr<Type>) {
    expr.transform(&mut |ref mut expr| {
        let mut sym_gen = SymbolGenerator::from_expression(expr);
        if let For{iters: ref all_iters, builder: ref outer_bldr, func: ref outer_func} = expr.kind {
            if all_iters.len() > 1 {
                // Vector of tuples containing the params and expressions of functions in nested lambdas.
                let mut lambdas = vec![];
                let mut common_data = None;
                // Used to check if the same rows of each output are touched by the outer for.
                let first_iter = (&all_iters[0].start, &all_iters[0].end, &all_iters[0].stride);
                // First, check if all the lambdas are over the same vector and have a pattern we can merge.
                // Below, for each iterator in the for loop, we checked if each nested for loop is
                // over the same vector and has the same Iter parameters (i.e., same start, end, stride).
                if all_iters.iter().all(|ref iter| {
                    if (&iter.start, &iter.end, &iter.stride) == first_iter {
                        // Make sure each nested for loop follows the ``result(for(a, appender, ...)) pattern.
                        if let Res{builder: ref res_bldr} = iter.data.kind {
                            if let For{iters: ref iters2, builder: ref bldr2, func: ref lambda} = res_bldr.kind {
                                if common_data.is_none() {
                                    common_data = Some(iters2.clone());
                                }
                                if iters2 == common_data.as_ref().unwrap() { 
                                    if let NewBuilder(_) = bldr2.kind {
                                        if let Builder(ref kind, _) = bldr2.ty {
                                            if let Appender(_) = *kind {
                                                if let Lambda{params: ref args, ref body} = lambda.kind {
                                                    if let Merge{ref builder, ref value} = body.kind {
                                                        if let Ident(ref n) = builder.kind {
                                                            if *n == args[0].name {
                                                                // Save the arguments and expressions for the function so
                                                                // they can be used for fusion later.
                                                                lambdas.push((args.clone(), value.clone())); 
                                                                return true
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // The pattern doesn't match for some Iter -- abort the transform.
                    return false
                }) {
                    // All Iters are over the same range and same vector, with a pattern we can
                    // transform. Produce the new expression by zipping the functions of each
                    // nested for into a single merge into a struct.
 
                    // Zip the expressions to create an appender whose merge (value) type is a struct.
                    let merge_type = Struct(lambdas.iter().map(|ref e| e.1.ty.clone()).collect::<Vec<_>>());
                    // TODO(Deepak): Fix this to something meaningful.
                    let builder_type = Builder(Appender(Box::new(merge_type.clone())), Annotations::new());
                    // The element type remains unchanged.
                    let func_elem_type = lambdas[0].0[2].ty.clone();

                    // Parameters for the new fused function. Symbols names are generated using symbol
                    // names for the builder and element from an existing function.
                    let new_params = vec![
                        Parameter{ty: builder_type.clone(), name: sym_gen.new_symbol(&lambdas[0].0[0].name.name)},
                        Parameter{ty: Scalar(ScalarKind::I64), name: sym_gen.new_symbol(&lambdas[0].0[1].name.name)},
                        Parameter{ty: func_elem_type.clone(), name: sym_gen.new_symbol(&lambdas[0].0[2].name.name)},
                    ];

                    // Generate Ident expressions for the new symbols and substitute them in the
                    // functions' merge expressions.
                    let new_bldr_expr = Expr {
                        ty: builder_type.clone(),
                        kind: Ident(new_params[0].name.clone()),
                        annotations: Annotations::new(),
                    };
                    let new_index_expr = Expr {
                        ty: Scalar(ScalarKind::I64),
                        kind: Ident(new_params[1].name.clone()),
                        annotations: Annotations::new(),
                    };
                    let new_elem_expr = Expr {
                        ty: func_elem_type.clone(),
                        kind: Ident(new_params[2].name.clone()),
                        annotations: Annotations::new(),
                    };
                    for &mut (ref mut args, ref mut expr) in lambdas.iter_mut() {
                        expr.substitute(&args[0].name, &new_bldr_expr);
                        expr.substitute(&args[1].name, &new_index_expr);
                        expr.substitute(&args[2].name, &new_elem_expr);
                    }

                    // Build up the new expression. The new expression merges structs into an
                    // appender, where each struct field is an expression which was merged into an
                    // appender in one of the original functions. For example, if there were two
                    // zipped fors in the original expression with lambdas |b1,e1| merge(b1,
                    // e1+1) and |b2,e2| merge(b2, e2+2), the new expression would be merge(b,
                    // {e+1,e+2}) into a new builder b of type appender[{i32,i32}]. e1, e2, and e
                    // refer to the same element in the expressions above since we check to ensure
                    // each zipped for is over the same input data.
                    let new_merge_expr = Expr{
                        ty: builder_type.clone(),
                        kind: Merge{
                            builder: Box::new(new_bldr_expr),
                            value: Box::new(Expr{
                                ty: merge_type.clone(),
                                kind: MakeStruct{elems: lambdas.iter().map(|ref lambda| *lambda.1.clone()).collect::<Vec<_>>()},
                                annotations: Annotations::new(),
                            })
                        },
                        annotations: Annotations::new(),
                    };
                    let new_func = Expr{
                        ty: Function(new_params.iter().map(|ref p| p.ty.clone()).collect::<Vec<_>>(), Box::new(builder_type.clone())),
                        kind: Lambda{params: new_params, body: Box::new(new_merge_expr)},
                        annotations: Annotations::new(),
                    };
                    let new_iter_expr = Expr{
                        ty: Vector(Box::new(merge_type.clone())),
                        kind: Res{builder: Box::new(Expr{
                            ty: builder_type.clone(),
                            kind: For{iters: common_data.unwrap(), builder: Box::new(Expr{ty: builder_type.clone(), kind: NewBuilder(None), annotations: Annotations::new()}), func: Box::new(new_func)},
                            annotations: Annotations::new(),
                        })},
                        annotations: Annotations::new(),
                    };

                    // TODO(shoumik): Any way to avoid the clones here?
                    return Some(Expr{
                        ty: expr.ty.clone(),
                        kind: For{iters: vec![Iter{
                            data: Box::new(new_iter_expr),
                            start: all_iters[0].start.clone(),
                            end: all_iters[0].end.clone(),
                            stride: all_iters[0].stride.clone()
                        }], builder: outer_bldr.clone(), func: outer_func.clone()},
                        annotations: Annotations::new(),
                    });
                }
            }
        }
        None
    });
}

/// Fuses loops where one for loop takes another as it's input, which prevents intermediate results
/// from being materialized.
pub fn fuse_loops_vertical(expr: &mut Expr<Type>) {
    expr.transform(&mut |ref mut expr| {
        let mut sym_gen = SymbolGenerator::from_expression(expr);
        if let For {
                   iters: ref all_iters,
                   builder: ref bldr1,
                   func: ref nested,
               } = expr.kind {
            if all_iters.len() == 1 {
                let ref iter1 = all_iters[0];
                if let Res { builder: ref res_bldr } = iter1.data.kind {
                    if let For {
                               iters: ref iters2,
                               builder: ref bldr2,
                               func: ref lambda,
                           } = res_bldr.kind {
                        if iters2.iter().all(|ref i| consumes_all(&i)) {
                            if let NewBuilder(_) = bldr2.kind {
                                if let Builder(ref kind, _) = bldr2.ty {
                                    if let Appender(_) = *kind {
                                        let e = Expr {
                                            ty: expr.ty.clone(),
                                            kind: For {
                                                iters: iters2.clone(),
                                                builder: bldr1.clone(),
                                                func: Box::new(replace_builder(lambda,
                                                                               nested,
                                                                               &mut sym_gen)),
                                            },
                                            annotations: Annotations::new(),
                                        };
                                        return Some(e);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    });
}

/// Given an iterator, returns whether the iterator consumes every element of its data vector.
fn consumes_all(iter: &Iter<Type>) -> bool {
    if let &Iter {
               start: None,
               end: None,
               stride: None,
               ..
           } = iter {
        return true;
    } else if let &Iter {
                      ref data,
                      start: Some(ref start),
                      end: Some(ref end),
                      stride: Some(ref stride),
                  } = iter {
        // Checks if the stride is 1 and an entire vector represented by a symbol is consumed.
        if let (&Literal(I64Literal(1)),
                &Literal(I64Literal(0)),
                &Ident(ref name),
                &Length { data: ref v }) = (&stride.kind, &start.kind, &data.kind, &end.kind) {
            if let Ident(ref vsym) = v.kind {
                return vsym == name;
            }
        }
        // Checks if an entire vector literal is consumed.
        if let (&Literal(I64Literal(1)), &Literal(I64Literal(0)), &MakeVector { ref elems }) =
            (&stride.kind, &start.kind, &data.kind) {
            let num_elems = elems.len() as i64;
            if let Literal(I64Literal(x)) = end.kind {
                return num_elems == x;
            }
        }
    }
    false
}

/// Given a lambda which takes a builder and an argument, returns a new function which takes a new
/// builder type and calls nested on the values it would've merged into its old builder. This
/// allows us to "compose" merge functions and avoid creating intermediate results.
fn replace_builder(lambda: &Expr<Type>,
                   nested: &Expr<Type>,
                   sym_gen: &mut SymbolGenerator)
                   -> Expr<Type> {

    // Tests whether an identifier and symbol refer to the same value by
    // comparing the symbols.
    fn same_iden(a: &ExprKind<Type>, b: &Symbol) -> bool {
        if let Ident(ref symbol) = *a {
            symbol == b
        } else {
            false
        }
    }

    let mut new_func = None;
    if let Lambda {
               params: ref args,
               ref body,
           } = lambda.kind {
        if let Lambda { params: ref nested_args, .. } = nested.kind {
            let mut new_body = *body.clone();
            let ref old_bldr = args[0];
            let ref old_index = args[1];
            let ref old_arg = args[2];
            let new_bldr_sym = sym_gen.new_symbol(&old_bldr.name.name);
            let new_index_sym = sym_gen.new_symbol(&old_index.name.name);
            let new_bldr = Expr {
                ty: nested_args[0].ty.clone(),
                kind: Ident(new_bldr_sym.clone()),
                annotations: Annotations::new(),
            };
            let new_index = Expr {
                ty: nested_args[1].ty.clone(),
                kind: Ident(new_index_sym.clone()),
                annotations: Annotations::new(),
            };
            new_body.ty = new_bldr.ty.clone();
            new_body.transform_and_continue(&mut |ref mut e| match e.kind {
                Merge { ref builder, ref value } if same_iden(&(*builder).kind, &old_bldr.name) => {
                    let params: Vec<Expr<Type>> =
                        vec![new_bldr.clone(), new_index.clone(), *value.clone()];
                    let mut expr = Expr {
                        ty: e.ty.clone(),
                        kind: Apply {
                            func: Box::new(nested.clone()),
                            params: params,
                        },
                        annotations: Annotations::new(),
                    };
                    inline_apply(&mut expr);
                    (Some(expr), true)
                }
                For { iters: ref data, builder: ref bldr, ref func }
                    if same_iden(&(*bldr).kind, &old_bldr.name) => {
                    (Some(Expr {
                         ty: e.ty.clone(),
                         kind: For {
                             iters: data.clone(),
                             builder: Box::new(new_bldr.clone()),
                             func: Box::new(replace_builder(func, nested, sym_gen)),
                         },
                         annotations: Annotations::new(),
                     }),
                     false)
                }
                Ident(ref mut symbol) if *symbol == old_bldr.name => {
                    (Some(new_bldr.clone()), false)
                }
                Ident(ref mut symbol) if *symbol == old_index.name => {
                    (Some(new_index.clone()), false)
                }
                _ => (None, true),
            });
            let new_params = vec![Parameter {
                                      ty: new_bldr.ty.clone(),
                                      name: new_bldr_sym.clone(),
                                  },
                                  Parameter {
                                      ty: Scalar(ScalarKind::I64),
                                      name: new_index_sym.clone(),
                                  },
                                  Parameter {
                                      ty: old_arg.ty.clone(),
                                      name: old_arg.name.clone(),
                                  }];

            if let Function(_, ref ret_ty) = nested.ty {
                let new_func_type =
                    Function(new_params.iter().map(|e| e.ty.clone()).collect::<Vec<_>>(),
                             ret_ty.clone());
                new_func = Some(Expr {
                                    ty: new_func_type,
                                    kind: Lambda {
                                        params: new_params,
                                        body: Box::new(new_body),
                                    },
                                    annotations: Annotations::new(),
                                })
            }
        }
    }

    if let Some(new) = new_func {
        new
    } else {
        nested.clone()
    }
}
