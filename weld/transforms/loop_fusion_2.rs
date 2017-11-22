//! Richer loop fusion rules.

use ast::*;
use ast::ExprKind::*;
use ast::Type::*;
use ast::BuilderKind::*;
use error::*;

use super::uniquify;

use util::SymbolGenerator;

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
pub fn fuse_loops_2(expr: &mut Expr<Type>) {
    use exprs::*;
    if uniquify::uniquify(expr).is_err() {
        return;
    }
    let mut gen = SymbolGenerator::from_expression(expr);
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

                        // If needed, add a Let statement to package the map_elems into a struct, and substitute
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
pub fn move_merge_before_let(expr: &mut Expr<Type>) {
    use exprs::*;
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

/// Checks whether a For loop is simple enough to be fused.
fn is_fusable_expr(expr: &TypedExpr) -> bool {
    if let Some(rfa) = ResForAppender::extract(expr) {
        if rfa.iters.iter().all(|ref i| i.is_simple()) {
            if let Some(_) = MergeSingle::extract(&rfa.func) {
                return true;
            }
        }
    }
    return false;
}

/// Checks if a name binding can be fused with the loop its contained in.
fn only_used_in_zip(name: &Symbol, expr: &TypedExpr) -> bool {
    // Number of times the name appears in `expr`.
    let mut total_count = 0;
    // Number of times the name appears in a Zip in `expr`.
    let mut iters_count = 0;
    expr.traverse(&mut |ref expr| {
        match expr.kind {
            Ident(ref name1) if name == name1 => {
                total_count += 1;
            }
            For { ref iters, .. } => {
                for iter in iters.iter() {
                    match iter.data.kind {
                        Ident(ref name1) if name == name1 => {
                            iters_count += 1;
                        }
                        _ => ()
                    }
                }
            }
            Length { ref data } => {
                if let Ident(ref name1) = data.kind {
                    if name1 == name {
                        total_count -= 1;
                    }
                }
            }
            _ => ()
        };
    });
    (iters_count == total_count)
}

/// Aggressively inlines let statements in cases which allow loop fusion to fire. This inliner is
/// aggressive because it will replace identifiers which appear more than once after being defined.
/// However, the inliner will only fire if eventually, the inlined loop will be fused.
pub fn aggressive_inline_let(expr: &mut TypedExpr) {
    let mut subbed_one = false;
    expr.transform_up(&mut |ref mut expr| {
        if subbed_one {
            return None;
        }
        if let Let { ref mut name, ref mut value, ref mut body } = expr.kind {
            if !is_fusable_expr(value) {
                return None;
            } else if !only_used_in_zip(name, body) {
                return None;
            }
            let mut new_body = body.as_ref().clone();
            new_body.substitute(name, value);
            subbed_one = true;
            Some(new_body)
        } else {
            None
        }
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
