//! Richer loop fusion rules.

use crate::ast::BuilderKind::*;
use crate::ast::ExprKind::*;
use crate::ast::Type::*;
use crate::ast::*;
use crate::error::*;

use crate::util::SymbolGenerator;

use fnv;

struct MergeSingle<'a> {
    params: &'a Vec<Parameter>,
    value: &'a Expr,
}

impl<'a> MergeSingle<'a> {
    fn extract(expr: &'a Expr) -> Option<MergeSingle<'a>> {
        if let Lambda {
            ref params,
            ref body,
        } = expr.kind
        {
            if let Merge {
                ref builder,
                ref value,
            } = body.kind
            {
                match builder.kind {
                    Ident(ref name) if *name == params[0].name => {
                        return Some(MergeSingle { params, value });
                    }
                    _ => {}
                }
            }
        }
        None
    }
}

struct NewAppender;

impl NewAppender {
    fn extract(expr: &Expr) -> Option<NewAppender> {
        if let NewBuilder(_) = expr.kind {
            if let Builder(Appender(_), _) = expr.ty {
                return Some(NewAppender);
            }
        }
        None
    }
}

struct ResForAppender<'a> {
    iters: &'a Vec<Iter>,
    func: &'a Expr,
}

impl<'a> ResForAppender<'a> {
    fn extract(expr: &'a Expr) -> Option<ResForAppender<'a>> {
        if let Res { ref builder } = expr.kind {
            if let For {
                ref iters,
                ref builder,
                ref func,
            } = builder.kind
            {
                if NewAppender::extract(builder).is_some() {
                    return Some(ResForAppender { iters, func });
                }
            }
        }
        None
    }
}

struct MapIter<'a> {
    iters: &'a Vec<Iter>,
    merge_params: &'a Vec<Parameter>,
    merge_value: &'a Expr,
}

impl<'a> MapIter<'a> {
    fn extract(iter: &'a Iter) -> Option<MapIter<'_>> {
        if iter.is_simple() {
            if let Some(rfa) = ResForAppender::extract(&iter.data) {
                if rfa.iters.iter().all(|ref i| i.is_simple()) {
                    if let Some(merge) = MergeSingle::extract(&rfa.func) {
                        return Some(MapIter {
                            iters: &rfa.iters,
                            merge_params: merge.params,
                            merge_value: merge.value,
                        });
                    }
                }
            }
        }
        None
    }
}

/// Gets rid of `result(for(d, appender, _))` expressions within the iterators in a loop, replacing them
/// with direct iteration over the data (e.g. `d` here).
///
/// Caveats:
///   - Like all Zip-based transforms, this function currently assumes that the output of each
///     expression in the Zip is the same length.
pub fn fuse_loops_2(expr: &mut Expr) {
    if expr.uniquify().is_err() {
        return;
    }
    let mut gen = SymbolGenerator::from_expression(expr);
    expr.transform(&mut |ref mut expr| {
        if let For {
            ref iters,
            ref builder,
            ref func,
        } = expr.kind
        {
            if let Lambda {
                ref params,
                ref body,
            } = func.kind
            {
                // Check whether at least one iterator is over a map pattern, i.e., Res(For(new Appender)) with
                // a simple merge operation.
                if !iters
                    .iter()
                    .any(|ref iter| MapIter::extract(iter).is_some())
                {
                    return None;
                }

                // Now that we know we have at least one nested map, create a new expression to replace the old
                // one with. We start by figuring out our new list of iterators, then build a new lambda function
                // that will call the old one but substitute some of the variables in it.
                let mut new_iters: Vec<Iter> = Vec::new();
                let mut new_elem_exprs: Vec<Expr> = Vec::new();
                let mut let_statements: Vec<(Symbol, Expr)> = Vec::new();
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
                        let index_ident =
                            Expr::new_ident(params[1].name.clone(), params[1].ty.clone()).unwrap();
                        value.substitute(&map.merge_params[1].name, &index_ident);

                        // For each iterator in the original MapIter, figure out a local variable that will hold
                        // the corresponding item.
                        let mut map_elem_types: Vec<Type> = Vec::new();
                        let mut map_elem_symbols: Vec<Symbol> = Vec::new();
                        let mut map_elem_exprs: Vec<Expr> = Vec::new();
                        for map_iter in map.iters.iter() {
                            // Check whether this iterator is already in our new_iters list; if so, reuse the old one
                            let iter_num;
                            if let Some(pos) = new_iters
                                .iter()
                                .position(|x| iters_match_ignoring_symbols(x, map_iter).unwrap())
                            {
                                iter_num = pos
                            } else {
                                // If it is indeed a new iterator, remember its element type and assign it a symbol.
                                new_iters.push((*map_iter).clone());
                                let elem_type = match map_iter.data.ty {
                                    Vector(ref ty) => ty,
                                    _ => panic!("Iterator was not over a vector"),
                                };
                                new_elem_types.push(elem_type.as_ref().clone());
                                let new_elem_symbol = gen.new_symbol("tmp");
                                new_elem_symbols.push(new_elem_symbol);
                                iter_num = new_iters.len() - 1;
                            }
                            let elem_ident = Expr::new_ident(
                                new_elem_symbols[iter_num].clone(),
                                new_elem_types[iter_num].clone(),
                            )
                            .unwrap();
                            map_elem_types.push(new_elem_types[iter_num].clone());
                            map_elem_symbols.push(new_elem_symbols[iter_num].clone());
                            map_elem_exprs.push(elem_ident);
                        }

                        // If needed, add a Let statement to package the map_elems into a struct, and substitute
                        // that into our value expression; otherwise just substitute the single symbol we're using
                        if map_elem_exprs.len() > 1 {
                            let struct_symbol = gen.new_symbol("tmp");
                            let make_struct = Expr::new_make_struct(map_elem_exprs).unwrap();
                            let struct_ident =
                                Expr::new_ident(struct_symbol.clone(), make_struct.ty.clone())
                                    .unwrap();
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
                        if let Some(pos) = new_iters
                            .iter()
                            .position(|x| iters_match_ignoring_symbols(x, &iters[i]).unwrap())
                        {
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
                        let elem_ident = Expr::new_ident(
                            new_elem_symbols[iter_num].clone(),
                            new_elem_types[iter_num].clone(),
                        )
                        .unwrap();
                        new_elem_exprs.push(elem_ident);
                    }
                }

                let new_param_type = if new_elem_types.len() > 1 {
                    Struct(new_elem_types.clone())
                } else {
                    new_elem_types[0].clone()
                };
                let new_param_name = gen.new_symbol("data");
                let new_param = Parameter {
                    name: new_param_name.clone(),
                    ty: new_param_type.clone(),
                };

                let new_params = vec![params[0].clone(), params[1].clone(), new_param];

                // Add a let statement in front of the body that builds up the argument struct.
                let old_param_expr = if new_elem_exprs.len() > 1 {
                    Expr::new_make_struct(new_elem_exprs).unwrap()
                } else {
                    new_elem_exprs[0].clone()
                };
                new_body = Expr::new_let(params[2].name.clone(), old_param_expr, new_body).unwrap();

                // Add any let statements we created for temporary structs.
                for pair in let_statements.iter().rev() {
                    new_body = Expr::new_let(pair.0.clone(), pair.1.clone(), new_body).unwrap()
                }

                // Add let statements in front of the body that set the new_elem_symbols to new_elem_exprs.
                let new_param_ident = Expr::new_ident(new_param_name, new_param_type).unwrap();
                if new_elem_types.len() > 1 {
                    for i in (0..new_elem_types.len()).rev() {
                        new_body = Expr::new_let(
                            new_elem_symbols[i].clone(),
                            Expr::new_get_field(new_param_ident.clone(), i as u32).unwrap(),
                            new_body,
                        )
                        .unwrap()
                    }
                } else {
                    new_body = Expr::new_let(new_elem_symbols[0].clone(), new_param_ident, new_body)
                        .unwrap()
                }

                let new_func = Expr::new_lambda(new_params, new_body).unwrap();
                let mut result =
                    Expr::new_for(new_iters, builder.as_ref().clone(), new_func).unwrap();
                result.annotations = expr.annotations.clone();
                return Some(result);
            }
        }

        None
    })
}

/// Replaces Let(name, value, Merge(builder, elem)) with Merge(builder, Let(name, value, elem)) to
/// enable further pattern matching on map functions downstream. This is only allowed when the let
/// statement is not defining some symbol that's used in the builder expression, so we check for that.
pub fn move_merge_before_let(expr: &mut Expr) {
    expr.transform_up(&mut |ref mut expr| {
        if let Let {
            ref name,
            value: ref let_value,
            ref body,
        } = expr.kind
        {
            if let Merge {
                ref builder,
                value: ref merge_value,
            } = body.kind
            {
                if !builder.contains_symbol(name) {
                    return Some(
                        Expr::new_merge(
                            *builder.clone(),
                            Expr::new_let(name.clone(), *let_value.clone(), *merge_value.clone())
                                .unwrap(),
                        )
                        .unwrap(),
                    );
                }
            }
        }
        None
    });
}

/// Checks whether a For loop is simple enough to be fused.
fn is_fusable_expr(expr: &Expr) -> bool {
    if let Some(rfa) = ResForAppender::extract(expr) {
        if rfa.iters.iter().all(|ref i| i.is_simple()) && MergeSingle::extract(&rfa.func).is_some()
        {
            return true;
        }
    }
    false
}

/// Checks if a name binding can be fused with the loop its contained in.
fn only_used_in_zip(name: &Symbol, expr: &Expr) -> bool {
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
                        _ => (),
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
            _ => (),
        };
    });
    (iters_count == total_count)
}

/// Aggressively inlines let statements in cases which allow loop fusion to fire. This inliner is
/// aggressive because it will replace identifiers which appear more than once after being defined.
/// However, the inliner will only fire if eventually, the inlined loop will be fused.
pub fn aggressive_inline_let(expr: &mut Expr) {
    let mut subbed_one = false;
    expr.transform_up(&mut |ref mut expr| {
        if subbed_one {
            return None;
        }
        if let Let {
            ref mut name,
            ref mut value,
            ref mut body,
        } = expr.kind
        {
            if !is_fusable_expr(value) || !only_used_in_zip(name, body) {
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

/// Merges { result(for(appender, result(for(appender( } over the same set of loops into
/// a single loop of result(for({appender,appender} ...)).
///
/// TODO this can definitely be generalized to capture more cases (e.g., the result doesn't
/// necessarily need to be in a `MakeStruct` expression).
///
/// Prerequisites: Expression is uniquified.
/// Caveats: This transformation will only fire if each vector in the iterator is bound to an
/// identifier.
pub fn merge_makestruct_loops(expr: &mut Expr) {
    expr.uniquify().unwrap();
    expr.transform(&mut |ref mut expr| {
        if let MakeStruct { ref elems } = expr.kind {
            // Each member of the `GetStruct` must be a ResForAppender pattern.
            if elems.len() > 2
                || !elems
                    .iter()
                    .all(|ref e| ResForAppender::extract(e).is_some())
            {
                return None;
            }
            let rfas: Vec<_> = elems
                .iter()
                .map(|ref e| ResForAppender::extract(e).unwrap())
                .collect();

            // Make sure all the iterators are simple, and each map just has a single merge.
            let all_iters_simple = rfas.iter().all(|ref rfa| {
                rfa.iters.iter().all(|ref iter| iter.is_simple())
                    && MergeSingle::extract(rfa.func).is_some()
            });

            if !all_iters_simple {
                return None;
            }

            // For each Iter, holds a map from name -> index. The indices are required to rewrite
            // struct accesses. Also keep a HashSet of the identifiers in each iterator.
            let mut ident_indices = vec![];
            let mut idents = vec![];

            // This is the "authoratative map", i.e., the GetField indexing every other body will be transformed
            // to use. It maps the *index to the name* (reverse of all the other maps).
            let mut first_rfa_map = fnv::FnvHashMap::default();

            for (i, rfa) in rfas.iter().enumerate() {
                let mut map = fnv::FnvHashMap::default();
                let mut set = fnv::FnvHashSet::default();
                for (j, iter) in rfa.iters.iter().enumerate() {
                    if let Ident(ref name) = iter.data.kind {
                        map.insert(j, name);
                        set.insert(name);
                        // Only for first one.
                        if i == 0 {
                            first_rfa_map.insert(name, j);
                        }
                    } else {
                        return None;
                    }
                }
                ident_indices.push(map);
                idents.push(set);
            }

            // For the future when we may support iteration over ranges.
            if idents.is_empty() {
                return None;
            }

            // Now, make sure each iterator has the same set of vectors (but perhaps in a different
            // ordering).
            if !idents.iter().all(|e| idents[0] == *e) {
                return None;
            }

            //
            // We now have a struct of ResForAppender patterns over the same data. Rewrite this to be a
            // ResForAppender over a struct of appender.
            //

            // Safe to unwrap since we checked it above.
            let first_ma = MergeSingle::extract(rfas[0].func).unwrap();

            // Construct the new builder type for the loop/etc.

            // For each RFA, get the first parameter of the builder function and copy its type.
            let types: Vec<_> = rfas
                .iter()
                .map(|ref rfa| MergeSingle::extract(rfa.func).unwrap().params[0].ty.clone())
                .collect();
            let final_builder_ty = Struct(types);

            let mut bodies = vec![];

            for (i, rfa) in rfas.iter().enumerate() {
                let ma = MergeSingle::extract(rfa.func).unwrap();
                let mut new_body = ma.value.clone();

                // If the element is a Struct that Zips multiple vectors, we need to rewrite the
                // indexing to match the first body. This handles zips over the same vectors in a
                // different order (which could be common if upstream transforms shuffle things around).
                if rfa.iters.len() > 1 {
                    let rev_map = &ident_indices[i];

                    // third parameter is the element.
                    let elem_name = &ma.params[2];

                    // This will be a no-op for the first iterator.
                    new_body.transform(&mut |ref mut e| {
                        if let GetField {
                            ref mut expr,
                            ref mut index,
                        } = e.kind
                        {
                            if let Ident(ref name) = expr.kind {
                                if *name == elem_name.name && expr.ty == elem_name.ty {
                                    // Get the vector identifier this index refers to.
                                    let vec_name = rev_map.get(&(*index as usize)).unwrap();
                                    let change_to = first_rfa_map[vec_name];
                                    *index = change_to as u32;
                                }
                            }
                        }
                        // Expression is modified in place.
                        None
                    });
                }

                // Substitute the parameter names to use the ones from the first body.
                // Skip the first one since that's the builder, and since this is matching a
                // MergeSingle pattern, we shouldn't have any builders in here.
                for (j, ref param) in ma.params.iter().enumerate() {
                    let replacement =
                        &Expr::new_ident(first_ma.params[j].name.clone(), param.ty.clone())
                            .unwrap();
                    new_body.substitute(&param.name, replacement);
                }
                // Add the new merge expression to the list of bodies.
                let builder_expr = Expr::new_get_field(
                    Expr::new_ident(first_ma.params[0].name.clone(), final_builder_ty.clone())
                        .unwrap(),
                    i as u32,
                )
                .unwrap();
                bodies.push(Expr::new_merge(builder_expr, new_body).unwrap());
            }

            let final_iters = rfas[0].iters.clone();
            let mut newbuilders = vec![];

            // Pull out the new builders and clone them into a vector.
            for elem in elems.iter() {
                if let Res { ref builder } = elem.kind {
                    if let For { ref builder, .. } = builder.kind {
                        newbuilders.push(builder.as_ref().clone());
                    }
                }
            }

            // Since we extracted RFAs from all of them...
            assert!(newbuilders.len() == elems.len());

            // Build the function and final body.
            let final_body = Expr::new_make_struct(bodies).unwrap();
            let mut final_params = first_ma.params.clone();
            final_params[0].ty = final_builder_ty.clone();

            let final_func = Expr::new_lambda(final_params, final_body).unwrap();
            let final_loop = Expr::new_for(
                final_iters,
                Expr::new_make_struct(newbuilders).unwrap(),
                final_func,
            )
            .unwrap();

            let mut gen = SymbolGenerator::from_expression(expr);
            let struct_name = gen.new_symbol("tmp");

            let builder_iden = Expr::new_ident(struct_name.clone(), final_builder_ty).unwrap();

            let results = (0..rfas.len())
                .map(|i| {
                    Expr::new_result(Expr::new_get_field(builder_iden.clone(), i as u32).unwrap())
                        .unwrap()
                })
                .collect();
            let results = Expr::new_make_struct(results).unwrap();
            let final_expr = Expr::new_let(struct_name, final_loop, results).unwrap();

            return Some(final_expr);
        }
        None
    });
}

/// Are two iterators equivalent ignoring symbols defined inside each one?
fn iters_match_ignoring_symbols(iter1: &Iter, iter2: &Iter) -> WeldResult<bool> {
    Ok(iter1.kind == iter2.kind
        && iter1.data.compare_ignoring_symbols(iter2.data.as_ref())?
        && options_match_ignoring_symbols(&iter1.start, &iter2.start)?
        && options_match_ignoring_symbols(&iter1.end, &iter2.end)?
        && options_match_ignoring_symbols(&iter1.stride, &iter2.stride)?)
}

/// Are two Option<Box<Expr>> equal ignoring symbols defined inside each one?
fn options_match_ignoring_symbols(
    opt1: &Option<Box<Expr>>,
    opt2: &Option<Box<Expr>>,
) -> WeldResult<bool> {
    match (opt1, opt2) {
        (&None, &None) => Ok(true),
        (&Some(ref e1), &Some(ref e2)) => e1.compare_ignoring_symbols(e2.as_ref()),
        _ => Ok(false),
    }
}
