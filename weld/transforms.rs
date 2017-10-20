//! Common transformations on expressions.

use super::ast::*;
use super::ast::ExprKind::*;
use super::ast::Type::*;
use super::ast::BuilderKind::*;
use super::ast::LiteralKind::*;
use super::error::*;
use super::exprs;

use std::collections::HashMap;

use super::util::SymbolGenerator;

/// Inlines Zip expressions as collections of iters. Using Zips outside of a For loop is currently
/// unsupported behavior. This transform handles the simple case of converting Zips in macros
/// such as map and filter into Iters in For loops.
///
/// TODO(shoumik): Perhaps Zip should just be a macro? Then macros need to be ordered.
pub fn inline_zips(expr: &mut Expr<Type>, _: &mut SymbolGenerator) {
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
                                     kind: first_iter.kind.clone(),
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
            *max_id += 1;
            if *id < *max_id {
                *id = *max_id;
            } else {
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
            pop_id(id_map, &name);
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
pub fn inline_apply<T: TypeBounds>(expr: &mut Expr<T>, _: &mut SymbolGenerator) {
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
pub fn inline_let(expr: &mut Expr<Type>, _: &mut SymbolGenerator) {
    if let Ok(_) = uniquify(expr) {
        expr.transform(&mut |ref mut expr| {
            if let Let {
                ref mut name,
                ref mut value,
                ref mut body,
            } = expr.kind {
                if symbol_usage_count(name, body) <= 1 {
                    body.transform(&mut |ref mut expr| {
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
pub fn fuse_loops_horizontal(expr: &mut Expr<Type>, gen: &mut SymbolGenerator) {
    expr.transform(&mut |ref mut expr| {
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
                        Parameter{ty: builder_type.clone(), name: gen.new_symbol(&lambdas[0].0[0].name.name)},
                        Parameter{ty: Scalar(ScalarKind::I64), name: gen.new_symbol(&lambdas[0].0[1].name.name)},
                        Parameter{ty: func_elem_type.clone(), name: gen.new_symbol(&lambdas[0].0[2].name.name)},
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
                            stride: all_iters[0].stride.clone(),
                            kind: all_iters[0].kind.clone(),
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
pub fn fuse_loops_vertical(expr: &mut Expr<Type>, gen: &mut SymbolGenerator) {
    expr.transform_and_continue_res(&mut |ref mut expr| {
        if let For { iters: ref all_iters, builder: ref bldr1, func: ref nested } = expr.kind {
            if all_iters.len() == 1 {
                let ref iter1 = all_iters[0];
                if let Res { builder: ref res_bldr } = iter1.data.kind {
                    if let For { iters: ref iters2, builder: ref bldr2, func: ref lambda, } = res_bldr.kind {
                        if iters2.iter().all(|ref i| consumes_all(&i)) {
                            if let NewBuilder(_) = bldr2.kind {
                                if let Builder(ref kind, _) = bldr2.ty {
                                    if let Appender(_) = *kind {
                                        let e = exprs::for_expr(iters2.clone(), *bldr1.clone(), replace_builder(lambda, nested, gen)?, false)?;
                                        return Ok((Some(e), true));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok((None, true))
    });
}

struct MergeSingle<'a> {
    params: &'a Vec<TypedParameter>,
    value: &'a TypedExpr
}

impl<'a> MergeSingle<'a> {
    fn extract(expr: &'a TypedExpr) -> Option<MergeSingle<'a>> {
        if let Lambda{ref params, ref body} = expr.kind {
            if let Merge{ref builder, ref value} = body.kind {
                match builder.kind {
                    Ident(ref name) if *name == params[0].name =>
                        return Some(MergeSingle{params, value}),
                    _ => {}
                }
            }
        }
        return None
    }
}

struct NewAppender<'a> {
    elem_type: &'a Type
}

impl<'a> NewAppender<'a> {
    fn extract(expr: &'a TypedExpr) -> Option<NewAppender<'a>> {
        if let NewBuilder(_) = expr.kind {
            if let Builder(Appender(ref elem_type), _) = expr.ty {
                return Some(NewAppender{elem_type})
            }
        }
        return None
    }
}

struct ResForAppender<'a> {
    iters: &'a Vec<Iter<Type>>,
    func: &'a TypedExpr
}

impl<'a> ResForAppender<'a> {
    fn extract(expr: &'a TypedExpr) -> Option<ResForAppender<'a>> {
        if let Res{ref builder} = expr.kind {
            if let For{ref iters, ref builder, ref func} = builder.kind {
                if NewAppender::extract(builder).is_some() {
                    return Some(ResForAppender{iters, func})
                }
            }
        }
        return None;
    }
}

struct MapIter<'a> {
    iters: &'a Vec<Iter<Type>>,
    merge_params: &'a Vec<TypedParameter>,
    merge_value: &'a TypedExpr
}

impl<'a> MapIter<'a> {
    fn extract(iter: &'a TypedIter) -> Option<MapIter> {
        if iter.is_simple() {
            if let Some(rfa) = ResForAppender::extract(&iter.data) {
                if rfa.iters.iter().all(|ref i| i.is_simple()) {
                    if let Some(merge) = MergeSingle::extract(&rfa.func) {
                        return Some(MapIter {
                            iters: &rfa.iters,
                            merge_params: merge.params,
                            merge_value: merge.value
                        });
                    }
                }
            }
        }
        return None;
    }
}

/// Gets rid of `result(for(d, appender, _))` expressions within the iterators in a loop, replacing them
/// with direct iteration over the data (e.g. `d` here).
///
/// Caveats:
///   - Like all Zip-based transforms, this function currently assumes that the output of each
///     expression in the Zip is the same length.
pub fn fuse_loops_2(expr: &mut Expr<Type>, gen: &mut SymbolGenerator) {
    use super::exprs::*;
    expr.transform(&mut |ref mut expr| {
        if let For{ref iters, ref builder, ref func} = expr.kind {
            if let Lambda{ref params, ref body} = func.kind {
                // Check whether at least one iterator is over a map pattern, i.e., Res(For(new Appender)) with
                // a simple merge operation.
                if !iters.iter().any(|ref iter| MapIter::extract(iter).is_some()) {
                    return None;
                }

                // Now that we know we have at least one nested map, create a new expression to replace the old
                // one with. We start by figuring out our new list of iterators, then build a new lambda function
                // that will call the old one but substitute some of the variables in it.
                let mut new_iters: Vec<TypedIter> = Vec::new();
                let mut new_elem_exprs: Vec<TypedExpr> = Vec::new();
                let mut let_statements: Vec<(Symbol, TypedExpr)> = Vec::new();
                let mut new_body = body.as_ref().clone();

                let elem_types: Vec<Type>;
                if let Struct(ref types) = params[2].ty {
                    elem_types = types.clone();
                } else {
                    elem_types = vec![params[2].ty.clone()];
                }
                let mut new_elem_types: Vec<Type> = Vec::new();
                let mut new_elem_symbols: Vec<Symbol> = Vec::new();

                for i in 0..iters.len() {
                    // Check whether this iter follows a map pattern, and if so, use that map's inner iter.
                    if let Some(ref map) = MapIter::extract(&iters[i]) {
                        // This was indeed a map pattern; we'll update it to apply the map function directly on the
                        // element values we pull from those iterators in the upper-level loop.
                        let mut value = map.merge_value.clone();
                        let index_ident = ident_expr(params[1].name.clone(), params[1].ty.clone()).unwrap();
                        value.substitute(&map.merge_params[1].name, &index_ident);

                        // For each iterator in the original MapIter, figure out a local variable that will hold
                        // the corresponding item.
                        let mut map_elem_types: Vec<Type> = Vec::new();
                        let mut map_elem_symbols: Vec<Symbol> = Vec::new();
                        let mut map_elem_exprs: Vec<TypedExpr> = Vec::new();
                        for ref map_iter in map.iters {
                            // Check whether this iterator is already in our new_iters list; if so, reuse the old one
                            let iter_num;
                            if let Some(pos) = new_iters.iter().position(
                                    |x| iters_match_ignoring_symbols(x, map_iter).unwrap()) {
                                iter_num = pos
                            } else {
                                // If it is indeed a new iterator, remember its element type and assign it a symbol.
                                new_iters.push((*map_iter).clone());
                                let elem_type = match &map_iter.data.ty {
                                    &Vector(ref ty) => ty,
                                    _ => panic!("Iterator was not over a vector")
                                };
                                new_elem_types.push(elem_type.as_ref().clone());
                                let new_elem_symbol = gen.new_symbol("tmp");
                                new_elem_symbols.push(new_elem_symbol);
                                iter_num = new_iters.len() - 1;
                            }
                            let elem_ident = ident_expr(
                                new_elem_symbols[iter_num].clone(), new_elem_types[iter_num].clone()).unwrap();
                            map_elem_types.push(new_elem_types[iter_num].clone());
                            map_elem_symbols.push(new_elem_symbols[iter_num].clone());
                            map_elem_exprs.push(elem_ident);
                        }

                        // If needed, add a Let statement to package the map_elems into a struct, and substitue
                        // that into our value expression; otherwise just substitute the single symbol we're using
                        if map_elem_exprs.len() > 1 {
                            let struct_symbol = gen.new_symbol("tmp");
                            let make_struct = makestruct_expr(map_elem_exprs).unwrap();
                            let struct_ident = ident_expr(struct_symbol.clone(), make_struct.ty.clone()).unwrap();
                            let_statements.push((struct_symbol, make_struct));
                            value.substitute(&map.merge_params[2].name, &struct_ident);
                        } else {
                            value.substitute(&map.merge_params[2].name, &map_elem_exprs[0]);
                        }

                        // Push an expression for this element
                        new_elem_exprs.push(value);
                    } else {
                        // Check whether this iterator is already in our new_iters list; if so, reuse the old one
                        let iter_num;
                        if let Some(pos) = new_iters.iter().position(
                                |x| iters_match_ignoring_symbols(x, &iters[i]).unwrap()) {
                            iter_num = pos
                        } else {
                            // If it is indeed a new iterator, remember its element type and assign it a symbol.
                            new_iters.push(iters[i].clone());
                            new_elem_types.push(elem_types[i].clone());
                            let new_elem_symbol = gen.new_symbol("tmp");
                            new_elem_symbols.push(new_elem_symbol);
                            iter_num = new_iters.len() - 1;
                        }
                        // Push an expression for this element.
                        let elem_ident = ident_expr(
                            new_elem_symbols[iter_num].clone(), new_elem_types[iter_num].clone()).unwrap();
                        new_elem_exprs.push(elem_ident);
                    }
                }

                let new_param_type = if new_elem_types.len() > 1 {
                    Struct(new_elem_types.clone())
                } else {
                    new_elem_types[0].clone()
                };
                let new_param_name = gen.new_symbol("data");
                let new_param = Parameter{name: new_param_name.clone(), ty: new_param_type.clone()};

                let new_params = vec![params[0].clone(), params[1].clone(), new_param];

                // Add a let statement in front of the body that builds up the argument struct.
                let old_param_expr = if new_elem_exprs.len() > 1 {
                    makestruct_expr(new_elem_exprs.clone()).unwrap()
                } else {
                    new_elem_exprs[0].clone()
                };
                new_body = let_expr(params[2].name.clone(), old_param_expr, new_body).unwrap();

                // Add any let statements we created for temporary structs.
                for pair in let_statements.iter().rev() {
                    new_body = let_expr(pair.0.clone(), pair.1.clone(), new_body).unwrap()
                }

                // Add let statements in front of the body that set the new_elem_symbols to new_elem_exprs.
                let new_param_ident = ident_expr(new_param_name.clone(), new_param_type.clone()).unwrap();
                if new_elem_types.len() > 1 {
                    for i in (0..new_elem_types.len()).rev() {
                        new_body = let_expr(
                            new_elem_symbols[i].clone(),
                            getfield_expr(new_param_ident.clone(), i as u32).unwrap(),
                            new_body
                        ).unwrap()
                    }
                } else {
                    new_body = let_expr(
                        new_elem_symbols[0].clone(),
                        new_param_ident.clone(),
                        new_body
                    ).unwrap()
                }

                let new_func = lambda_expr(new_params, new_body).unwrap();
                let result = for_expr(new_iters.clone(), builder.as_ref().clone(), new_func, false).unwrap();
                return Some(result);
            }
        }

        None
    })
}

/// Replaces Let(name, value, Merge(builder, elem)) with Merge(builder, Let(name, value, elem)) to
/// enable further pattern matching on map functions downstream. This is only allowed when the let
/// statement is not defining some symbol that's used in the builder expression, so we check for that.
pub fn move_merge_before_let(expr: &mut Expr<Type>, _: &mut SymbolGenerator) {
    use super::exprs::*;
    expr.transform_up(&mut |ref mut expr| {
        if let Let { ref name, value: ref let_value, ref body } = expr.kind {
            if let Merge { ref builder, value: ref merge_value } = body.kind {
                if !builder.contains_symbol(name) {
                    return Some(merge_expr(
                        *builder.clone(),
                        let_expr(name.clone(), *let_value.clone(), *merge_value.clone()).unwrap()
                    ).unwrap());
                }
            }
        }
        return None;
    });
}

/// Are two iterators equivalent ignoring symbols defined inside each one?
fn iters_match_ignoring_symbols(iter1: &TypedIter, iter2: &TypedIter) -> WeldResult<bool> {
    Ok(iter1.kind == iter2.kind &&
        iter1.data.compare_ignoring_symbols(iter2.data.as_ref())? &&
        options_match_ignoring_symbols(&iter1.start, &iter2.start)? &&
        options_match_ignoring_symbols(&iter1.end, &iter2.end)? &&
        options_match_ignoring_symbols(&iter1.stride, &iter2.stride)?)
}

/// Are two Option<Box<Expr>> equal ignoring symbols defined inside each one?
fn options_match_ignoring_symbols(opt1: &Option<Box<TypedExpr>>, opt2: &Option<Box<TypedExpr>>) -> WeldResult<bool> {
    match (opt1, opt2) {
        (&None, &None) => Ok(true),
        (&Some(ref e1), &Some(ref e2)) => e1.compare_ignoring_symbols(e2.as_ref()),
        _ => Ok(false)
    }
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
                      ..
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
                   -> WeldResult<Expr<Type>> {

    // Tests whether an identifier and symbol refer to the same value by
    // comparing the symbols.
    fn same_iden(a: &ExprKind<Type>, b: &Symbol) -> bool {
        if let Ident(ref symbol) = *a {
            symbol == b
        } else {
            false
        }
    }

    if let Lambda { params: ref args, ref body } = lambda.kind {
        if let Lambda { params: ref nested_args, .. } = nested.kind {
            let mut new_body = *body.clone();
            let ref old_bldr = args[0];
            let ref old_index = args[1];
            let ref old_arg = args[2];
            let new_bldr_sym = sym_gen.new_symbol(&old_bldr.name.name);
            let new_index_sym = sym_gen.new_symbol(&old_index.name.name);
            let new_bldr = exprs::ident_expr(new_bldr_sym.clone(), nested_args[0].ty.clone())?;
            let new_index = exprs::ident_expr(new_index_sym.clone(), nested_args[1].ty.clone())?;

            // Fix expressions to use the new builder.
            new_body.transform_and_continue_res(&mut |ref mut e| match e.kind {
                Merge { ref builder, ref value } if same_iden(&(*builder).kind, &old_bldr.name) => {
                    let params: Vec<Expr<Type>> = vec![new_bldr.clone(), new_index.clone(), *value.clone()];
                    let mut expr = exprs::apply_expr(nested.clone(), params)?;
                    inline_apply(&mut expr, sym_gen);
                    Ok((Some(expr), true))
                }
                For { iters: ref data, builder: ref bldr, ref func } if same_iden(&(*bldr).kind, &old_bldr.name) => {
                    let expr = exprs::for_expr(data.clone(), new_bldr.clone(), replace_builder(func, nested, sym_gen)?, false)?;
                    Ok((Some(expr), false))
                }
                Ident(ref mut symbol) if *symbol == old_bldr.name => {
                    Ok((Some(new_bldr.clone()), false))
                }
                Ident(ref mut symbol) if *symbol == old_index.name => {
                    Ok((Some(new_index.clone()), false))
                }
                _ => Ok((None, true)),
            });

            // Fix types to make sure the return type propagates through all subexpressions.
            match_types(&new_bldr.ty, &mut new_body);

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
            return exprs::lambda_expr(new_params, new_body);
        }
    }
    return weld_err!("Inconsistency in replace_builder");
}

/// Given a root type, forces each expression to return that type. TODO For now, only supporting
/// expressions which can be builders. We might want to factor this out to be somewhere else.
fn match_types(root_ty: &Type, expr: &mut Expr<Type>) {
    expr.ty = root_ty.clone();
    match expr.kind {
        If { ref mut on_true, ref mut on_false, ..} => {
            match_types(root_ty, on_true);
            match_types(root_ty, on_false);
        }
        Select { ref mut on_true, ref mut on_false, ..} => {
            match_types(root_ty, on_true);
            match_types(root_ty, on_false);
        }
        Let { ref mut body, ..} => {
            match_types(root_ty, body);
        }
        _ => {}
    };
}

/// Simplifies GetField(MakeStruct(*)) expressions, which can occur during loop fusion when some
/// of the loops are zipping together multiple column vectors.
pub fn simplify_get_field<T: TypeBounds>(expr: &mut Expr<T>, _: &mut SymbolGenerator) {
    expr.transform(&mut |ref mut expr| {
        if let GetField { ref expr, index } = expr.kind {
            if let MakeStruct { ref elems } = expr.kind {
                return Some(elems[index as usize].clone());
            }
        }
        None
    });
}

/// Infers the size of an `Appender` in a `For` loop.
pub fn infer_size(expr: &mut Expr<Type>, _: &mut SymbolGenerator) {
    expr.transform_and_continue_res(&mut |ref mut expr| {
        if let For { ref iters, ref builder, ref func } = expr.kind {
            // This constraint prevents copying expensive iters.
            if let Ident(_) = iters[0].data.kind {
                if let NewBuilder(None) = builder.kind {
                    if let Builder(Appender(ref ek), _) = builder.ty {
                        if let Lambda {ref params, ref body } = func.kind {
                            let ref builder_symbol = params[0].name.clone();
                            if simple_merge(builder_symbol, body) {
                                // Compute the inferred length based on the iter.
                                let length = if let Some(ref start) = iters[0].start {
                                    let e = exprs::binop_expr(BinOpKind::Subtract,
                                                              *iters[0].end.as_ref().unwrap().clone(),
                                                              *start.clone())?;
                                    exprs::binop_expr(BinOpKind::Divide, e, *iters[0].stride.as_ref().unwrap().clone())?
                                } else {
                                    exprs::length_expr(*iters[0].data.clone())?
                                };

                                let new_loop = exprs::for_expr(
                                    iters.clone(),
                                    exprs::newbuilder_expr(Appender(ek.clone()), Some(length))?,
                                    func.as_ref().clone(),
                                    false)?;
                                return Ok((Some(new_loop), false));
                            }
                        }
                    }
                }
            }
        }
        Ok((None, true))
    })
}

/// Checks that `expr` performs only one `Merge` per control path - this guarantees
/// that the loop containing `expr`'s size can be inferred.
fn simple_merge(sym: &Symbol, expr: &Expr<Type>) -> bool {
    match expr.kind {
        Merge { ref builder, ref value } => {
            if let Ident(ref s) = builder.kind {
                if s == sym {
                    return !value.contains_symbol(sym);
                }
            }
            return false;
        }
        If { ref cond, ref on_true, ref on_false } => {
            !cond.contains_symbol(sym) && simple_merge(sym, on_true) &&
                simple_merge(sym, on_false)
        }
        _ => false,
    }
}

/// Forces parallelization of for loops inside an `Iterate`. In the current LLVM code generator,
/// this prevents for loop continuations from making recursive calls to the beginning of the
/// iteration, which can cause stack exhaustion.
///
/// FIXME we need a long term solution to this!
pub fn force_iterate_parallel_fors(expr: &mut Expr<Type>, _: &mut SymbolGenerator) {
    expr.transform_and_continue(&mut |ref mut e| {
        match e.kind {
             Iterate { .. } => {
                e.transform_and_continue(&mut |ref mut e| {
                   match e.kind {
                        For { .. } => {
                            e.annotations.set_always_use_runtime(true);
                            (None, true)
                        }
                        _ => (None, true)
                   }
                });
                (None, false)
            }
            _ => {
                (None, true)
            }
        }
    });
}
