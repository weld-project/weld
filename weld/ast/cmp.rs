//! Compare expression trees ignoring Symbol values.

extern crate fnv;

use super::ast::*;
use error::*;

#[cfg(test)]
use tests::*;

/// A trait for symbol-agnostic tree comparison.
pub trait CompareIgnoringSymbols {

    /// Compare this expression to `other`.
    ///
    /// Returns whether the two expressions are the same modulo symbol names.
    /// Returns an error if an undefined symbol is encountered.
    fn compare_ignoring_symbols(&self, other: &Self) -> WeldResult<bool>;
}

impl CompareIgnoringSymbols for Expr {
    /// Compare this expression to `other`.
    fn compare_ignoring_symbols(&self, other: &Expr) -> WeldResult<bool> {
        use self::ExprKind::*;
        use fnv::FnvHashMap;
        let mut sym_map: FnvHashMap<&Symbol, &Symbol> = FnvHashMap::default();
        let mut reverse_sym_map: FnvHashMap<&Symbol, &Symbol> = FnvHashMap::default();
        fn _compare_ignoring_symbols<'b, 'a>(e1: &'a Expr,
                                             e2: &'b Expr,
                                             sym_map: &mut FnvHashMap<&'a Symbol, &'b Symbol>,
                                             reverse_sym_map: &mut FnvHashMap<&'b Symbol, &'a Symbol>)
            -> WeldResult<bool> {
            // First, check the type.
            if e1.ty != e2.ty {
                return Ok(false);
            }
            // Check the kind of each expression. same_kind is true if each *non-expression* field
            // is equal and the kind of the expression matches. Also records corresponding symbol names.
            let same_kind = match (&e1.kind, &e2.kind) {
                (&BinOp { kind: ref kind1, .. }, &BinOp { kind: ref kind2, .. }) if kind1 ==
                                                                                    kind2 => {
                    Ok(true)
                }
                (&UnaryOp { .. }, &UnaryOp { .. }) => Ok(true),
                (&Cast { kind: ref kind1, .. }, &Cast { kind: ref kind2, .. }) if kind1 ==
                                                                                  kind2 => Ok(true),
                (&ToVec { .. }, &ToVec { .. }) => Ok(true),
                (&Let { name: ref sym1, .. }, &Let { name: ref sym2, .. }) => {
                    sym_map.insert(sym1, sym2);
                    reverse_sym_map.insert(sym2, sym1);
                    Ok(true)
                }
                (&Lambda { params: ref params1, .. }, &Lambda { params: ref params2, .. }) => {
                    // Just compare types, and assume the symbol names "match up".
                    if params1.len() == params2.len() &&
                       params1.iter().zip(params2).all(|t| t.0.ty == t.1.ty) {
                        for (p1, p2) in params1.iter().zip(params2) {
                            sym_map.insert(&p1.name, &p2.name);
                            reverse_sym_map.insert(&p2.name, &p1.name);
                        }
                        Ok(true)
                    } else {
                        Ok(false)
                    }
                }
                (&NewBuilder(_), &NewBuilder(_)) => Ok(true),
                (&Negate(_), &Negate(_)) => Ok(true),
                (&Not(_), &Not(_)) => Ok(true),
                (&Broadcast(_), &Broadcast(_)) => Ok(true),
                (&MakeStruct { .. }, &MakeStruct { .. }) => Ok(true),
                (&MakeVector { .. }, &MakeVector { .. }) => Ok(true),
                (&Zip { .. }, &Zip { .. }) => Ok(true),
                (&GetField { index: idx1, .. }, &GetField { index: idx2, .. }) if idx1 == idx2 => {
                    Ok(true)
                }
                (&Length { .. }, &Length { .. }) => Ok(true),
                (&Lookup { .. }, &Lookup { .. }) => Ok(true),
                (&OptLookup { .. }, &OptLookup { .. }) => Ok(true),
                (&KeyExists { .. }, &KeyExists { .. }) => Ok(true),
                (&Slice { .. }, &Slice { .. }) => Ok(true),
                (&Sort { .. }, &Sort { .. }) => Ok(true),
                (&Merge { .. }, &Merge { .. }) => Ok(true),
                (&Res { .. }, &Res { .. }) => Ok(true),
                (&For { iters: ref liters, .. }, &For { iters: ref riters, .. })  => {
                    // If the iter kinds match for each iterator, the non-expression fields match.
                    Ok(liters.iter().zip(riters.iter()).all(|(ref l, ref r)| l.kind == r.kind))
                },
                (&If { .. }, &If { .. }) => Ok(true),
                (&Iterate { .. }, &Iterate { .. }) => Ok(true),
                (&Select { .. }, &Select { .. }) => Ok(true),
                (&Apply { .. }, &Apply { .. }) => Ok(true),
                (&CUDF {
                     sym_name: ref sym_name1,
                     return_ty: ref return_ty1,
                     ..
                 },
                 &CUDF {
                     sym_name: ref sym_name2,
                     return_ty: ref return_ty2,
                     ..
                 }) => {
                    let mut matches = sym_name1 == sym_name2;
                    matches = matches && return_ty1 == return_ty2;
                    Ok(matches)
                }
                (&Serialize(_), &Serialize(_)) => Ok(true),
                (&Deserialize { ref value_ty, .. }, &Deserialize { value_ty: ref value_ty2, .. }) if value_ty == value_ty2 => Ok(true),
                (&Literal(ref l), &Literal(ref r)) if l == r => Ok(true),
                (&Ident(ref l), &Ident(ref r)) => {
                    if let Some(lv) = sym_map.get(l) {
                        Ok(**lv == *r)
                    } else if reverse_sym_map.contains_key(r) {
                        Ok(false) // r was defined in other expression but l wasn't defined in self.
                    } else {
                        Ok(*l == *r)
                    }
                }
                _ => Ok(false), // all else fail.
            };

            // Return if encountered and error or kind doesn't match.
            if same_kind.is_err() || !same_kind.as_ref().unwrap() {
                return same_kind;
            }

            // Recursively check the children.
            let e1_children: Vec<_> = e1.children().collect();
            let e2_children: Vec<_> = e2.children().collect();
            if e1_children.len() != e2_children.len() {
                return Ok(false);
            }
            for (c1, c2) in e1_children.iter().zip(e2_children) {
                let res = _compare_ignoring_symbols(&c1, &c2, sym_map, reverse_sym_map);
                if res.is_err() || !res.as_ref().unwrap() {
                    return res;
                }
            }
            return Ok(true);
        }
        _compare_ignoring_symbols(self, other, &mut sym_map, &mut reverse_sym_map)
    }
}

#[test]
fn compare_expressions() {
    let e1 = parse_expr("for([1,2], appender, |e| e+1)").unwrap();
    let e2 = parse_expr("for([1,2], appender, |f| f+1)").unwrap();
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    let e1 = parse_expr("let a = 2; a").unwrap();
    let e2 = parse_expr("let b = 2; b").unwrap();
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    let e2 = parse_expr("let b = 2; c").unwrap();
    assert!(!e1.compare_ignoring_symbols(&e2).unwrap());

    // Undefined symbols should work as long as the symbol is the same.
    let e1 = parse_expr("[1, 2, 3, d]").unwrap();
    let e2 = parse_expr("[1, 2, 3, d]").unwrap();
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    let e2 = parse_expr("[1, 2, 3, e]").unwrap();
    assert!(!e1.compare_ignoring_symbols(&e2).unwrap());

    // Symbols can be substituted, so equal.
    let e1 = parse_expr("|a, b| a + b").unwrap();
    let e2 = parse_expr("|c, d| c + d").unwrap();
    assert!(e1.compare_ignoring_symbols(&e2).unwrap());

    // Symbols don't match up.
    let e2 = parse_expr("|c, d| d + c").unwrap();
    assert!(!e1.compare_ignoring_symbols(&e2).unwrap());
}
