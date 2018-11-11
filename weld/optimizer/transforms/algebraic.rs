
extern crate num_integer;

use ast::ExprKind::*;
use ast::LiteralKind::*;
use ast::Type::*;
use ast::*;
use ast::constructors::*;

use self::num_integer::Integer;

pub fn eliminate_redundant_negation(expr: &mut Expr) {
    expr.transform_kind(&mut eliminate_redundant_negation_impl)
}

fn eliminate_redundant_negation_impl(expr: &mut Expr) -> Option<ExprKind> {
    match expr.kind {
        Negate(ref mut outer) => {
            match outer.as_mut().kind {
                Negate(ref mut inner) => Some(inner.take().kind), // --x = x
                _ => None,
            }
        }
        BinOp {
            kind: binop_kind,
            left: ref mut lhs,
            right: ref mut rhs,
        } => {
            use ast::BinOpKind::*;
            match binop_kind {
                Subtract => {
                    match rhs.as_mut().kind {
                        Negate(ref mut inner) => {
                            // x - -y = x + y
                            let new = BinOp {
                                kind: BinOpKind::Add,
                                left: lhs.take(),
                                right: inner.take(),
                            };
                            Some(new)
                        }
                        _ => None,
                    }
                }
                Equal => {
                    // Awkward workaround for https://github.com/rust-lang/rfcs/issues/811
                    let lhs_is_not = match lhs.kind {
                        Not(_) => true,
                        _ => false,
                    };
                    let rhs_is_not = match rhs.kind {
                        Not(_) => true,
                        _ => false,
                    };
                    match (lhs_is_not, rhs_is_not) {
                        (true, true) => {
                            // !x == !y <=> x == y
                            if let Not(ref mut lhs_inner) = lhs.as_mut().kind {
                                if let Not(ref mut rhs_inner) = rhs.as_mut().kind {
                                    Some(BinOp {
                                        kind: Equal,
                                        left: lhs_inner.take(),
                                        right: rhs_inner.take(),
                                    })
                                } else {
                                    unreachable!();
                                }
                            } else {
                                unreachable!();
                            }
                        }
                        (true, false) => {
                            // !x == y <=> x != y
                            if let Not(ref mut lhs_inner) = lhs.as_mut().kind {
                                Some(BinOp {
                                    kind: NotEqual,
                                    left: lhs_inner.take(),
                                    right: rhs.take(),
                                })
                            } else {
                                unreachable!();
                            }
                        }
                        (false, true) => {
                            // x == !y <=> x != y
                            if let Not(ref mut rhs_inner) = rhs.as_mut().kind {
                                Some(BinOp {
                                    kind: NotEqual,
                                    left: lhs.take(),
                                    right: rhs_inner.take(),
                                })
                            } else {
                                unreachable!();
                            }
                        }
                        (false, false) => None,
                    }
                }
                NotEqual => {
                    // Awkward workaround for https://github.com/rust-lang/rfcs/issues/811
                    let lhs_is_not = match lhs.kind {
                        Not(_) => true,
                        _ => false,
                    };
                    let rhs_is_not = match rhs.kind {
                        Not(_) => true,
                        _ => false,
                    };
                    match (lhs_is_not, rhs_is_not) {
                        (true, true) => {
                            // !x != !y <=> x != y
                            if let Not(ref mut lhs_inner) = lhs.as_mut().kind {
                                if let Not(ref mut rhs_inner) = rhs.as_mut().kind {
                                    Some(BinOp {
                                        kind: NotEqual,
                                        left: lhs_inner.take(),
                                        right: rhs_inner.take(),
                                    })
                                } else {
                                    unreachable!();
                                }
                            } else {
                                unreachable!();
                            }
                        }
                        (true, false) => {
                            // !x != y <=> x == y
                            if let Not(ref mut lhs_inner) = lhs.as_mut().kind {
                                Some(BinOp {
                                    kind: Equal,
                                    left: lhs_inner.take(),
                                    right: rhs.take(),
                                })
                            } else {
                                unreachable!();
                            }
                        }
                        (false, true) => {
                            // x != !y <=> x == y
                            if let Not(ref mut rhs_inner) = rhs.as_mut().kind {
                                Some(BinOp {
                                    kind: Equal,
                                    left: lhs.take(),
                                    right: rhs_inner.take(),
                                })
                            } else {
                                unreachable!();
                            }
                        }
                        (false, false) => None,
                    }
                }
                _ => None,
            }
        }
        Not(ref mut outer) => {
            match outer.as_mut().kind {
                Literal(ref kind) => {
                    match kind {
                         // !true = false and !false = true
                        BoolLiteral(b) => Some(Literal(BoolLiteral(!b))),
                        _ => None,
                    }
                }
                Not(ref mut inner) => Some(inner.take().kind), // !!x = x
                BinOp {
                    kind: ref binop_kind,
                    left: ref mut lhs,
                    right: ref mut rhs,
                } => {
                    use ast::BinOpKind::*;
                    match binop_kind {
                        LogicalAnd | LogicalOr => {
                            /*
                             * !(e1 && e1) <=> !e1 || !e2 and !(e1 || e2) <=> !e1 && !e2
                             * This change is not a contraction, so it's a bit risky.
                             * The hope is that it allows eliminations further down the AST.
                             * But if not, it might actually increase the code size.
                             */
                            match (&lhs.kind, &rhs.kind) {
                                (&Ident(_), &Ident(_)) => None, // nothing to gain here
                                (_, _) => {
                                    let kind = flip_logical(binop_kind);
                                    Some(BinOp {
                                        kind,
                                        left: Box::new(not_expr(lhs.as_mut().take()).unwrap()),
                                        right: Box::new(not_expr(rhs.as_mut().take()).unwrap()),
                                    })
                                }
                            }
                        }
                        Equal => Some(BinOp {
                            // !(a == b) <=> a != b
                            kind: NotEqual,
                            left: lhs.take(),
                            right: rhs.take(),
                        }),
                        NotEqual => Some(BinOp {
                            // !(a != b) <=> a == b
                            kind: Equal,
                            left: lhs.take(),
                            right: rhs.take(),
                        }),
                        _ => None,
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Changes the associativity of expressions to compute operations over constants first.
///
/// This enables constant folding in downstream transformations.
pub fn shift_work_to_constants(expr: &mut Expr) {
    expr.transform_kind(&mut shift_work_to_constants_impl)
}

fn shift_work_to_constants_impl(expr: &mut Expr) -> Option<ExprKind> {
    match expr.kind {
        BinOp {
            kind: ref mut binop_kind,
            left: ref mut lhs,
            right: ref mut rhs,
        } => {
            match (is_constant(lhs), is_constant(rhs)) {
                (true, true) | (false, false) => None,
                (true, false) => {
                    match rhs.as_mut().kind {
                        BinOp {
                            kind: ref mut inner_binop_kind,
                            left: ref mut inner_lhs,
                            right: ref mut inner_rhs,
                        } => {
                            use self::ExprCmp::*;
                            use ast::BinOpKind::*;
                            let should_isolate = faster_change(inner_lhs, inner_rhs);
                            match (binop_kind, inner_binop_kind, should_isolate) {
                                (Add, Add, Left) => Some(
                                    // x + (y + z) = y + (x + z)
                                    right_associate(Add, inner_lhs.take(), lhs.take(), inner_rhs.take()),
                                ),
                                (Add, Add, Right) => Some(
                                    // x + (y + z) = (x + y) + z
                                    left_associate(Add, lhs.take(), inner_lhs.take(), inner_rhs.take()),
                                ),
                                (Add, Subtract, Left) => Some(
                                    // x + (y - z) = y + (x - z)
                                    right_associate2(Add, Subtract, inner_lhs.take(), lhs.take(), inner_rhs.take()),
                                ),
                                (Add, Subtract, Right) => Some(
                                    // x + (y - z) = (x + y) - z
                                    left_associate2(Add, Subtract, lhs.take(), inner_lhs.take(), inner_rhs.take()),
                                ),
                                //(Add, Multiply, _) => None, // x + (y * z)
                                //(Add, Divide, _) => None, // x + (y / z)
                                //(Add, Modulo, _) => None, // x + (y % z)
                                // (Add, ref cmp, _) => None // x + (y cmp z)
                                (Subtract, Add, Left) => Some(
                                    // x - (y + z) = (x - z) - y
                                    left_associate(Subtract, lhs.take(), inner_rhs.take(), inner_lhs.take()),
                                ),
                                (Subtract, Add, Right) => Some(
                                    // x - (y + z) = (x - y) - z
                                    left_associate(Subtract, lhs.take(), inner_lhs.take(), inner_rhs.take()),
                                ),
                                (Subtract, Subtract, Left) => Some(
                                    // x - (y - z) = (x + z) - y
                                    left_associate2(Add, Subtract, lhs.take(), inner_rhs.take(), inner_lhs.take()),
                                ),
                                (Subtract, Subtract, Right) => Some(
                                    // x - (y - z) =  (x - y) + z
                                    left_associate2(Subtract, Add, lhs.take(), inner_lhs.take(), inner_rhs.take()),
                                ),
                                //(Subtract, Multiply, _) => None, // x - (y * z)
                                //(Subtract, Divide, _) => None, // x - (y / z)
                                //(Subtract, Modulo, _) => None, // x - (y % z)
                                //(Multiply, Add, _) => None, // x * (y + z) = xy - xz
                                //(Multiply, Subtract, _) => None, // x * (y - z) = xy - xz
                                (Multiply, Multiply, Left) => Some(
                                    // x * (y * z) = y * (x * z)
                                    right_associate(Multiply, inner_lhs.take(), lhs.take(), inner_rhs.take()),
                                ),
                                (Multiply, Multiply, Right) => Some(
                                    // x * (y * z) = (x * y) * z
                                    left_associate(Multiply, lhs.take(), inner_lhs.take(), inner_rhs.take()),
                                ),
                                (Multiply, Divide, Left) => Some(
                                    // x * (y / z) = y * (x / z)
                                    right_associate2(Multiply, Divide, inner_lhs.take(), lhs.take(), inner_rhs.take()),
                                ),
                                (Multiply, Divide, Right) => Some(
                                    // x * (y / z) = (x * y) / z
                                    left_associate2(Multiply, Divide, lhs.take(), inner_lhs.take(), inner_rhs.take()),
                                ),
                                // (Multiply, Modulo, _) => None // x * (y % z)
                                //(Divide, Add, _) => None, // x / (y + z)
                                //(Divide, Subtract, _) => None, // x / (y - z)
                                (Divide, Multiply, Left) => Some(
                                    // x / (y * z) = (x / z) / y
                                    left_associate(Divide, lhs.take(), inner_rhs.take(), inner_lhs.take()),
                                ),
                                (Divide, Multiply, Right) => Some(
                                    // x / (y * z) = (x / y) / z
                                    left_associate(Divide, lhs.take(), inner_lhs.take(), inner_rhs.take()),
                                ),
                                (Divide, Divide, Left) => {
                                    // x / (y / z) = (x * z) / y
                                    use self::RangeClassification::*;
                                    match classify(inner_rhs) {
                                        Positive | Negative => {
                                            Some(left_associate2(Multiply, Divide, lhs.take(), inner_rhs.take(), inner_lhs.take()))
                                        }
                                        Zero | Unknown => None, // don't reorder on z=0, to avoid changing the division by 0 behaviour.
                                    }
                                }
                                (Divide, Divide, Right) => Some(
                                    // x / (y / z) = (x / y) / z
                                    left_associate(Divide, lhs.take(), inner_lhs.take(), inner_rhs.take()),
                                ),
                                // (Divide, Modulo, _) => None // x / (y % z)
                                (ref cmp, Add, Left) if cmp.is_comparison() => Some(
                                    // x cmp (y + z) <=> (x - z) flip(cmp) y
                                    left_associate2(Subtract, flip_comp(*cmp), lhs.take(), inner_rhs.take(), inner_lhs.take()),
                                ),
                                (ref cmp, Add, Right) if cmp.is_comparison() => Some(
                                    // x cmp (y + z) <=> (x - y) flip(cmp) z
                                    left_associate2(Subtract, flip_comp(*cmp), lhs.take(), inner_lhs.take(), inner_rhs.take()),
                                ),
                                (ref cmp, Subtract, Left) if cmp.is_comparison() => Some(
                                    // x cmp (y - z) <=> (x + z) cmp y
                                    left_associate2(Add, **cmp, lhs.take(), inner_rhs.take(), inner_lhs.take()),
                                ),
                                (ref cmp, Subtract, Right) if cmp.is_comparison() => Some(
                                    // x cmp (y - z) <=> z flip(cmp) (y - x)
                                    right_associate2(flip_comp(*cmp), Subtract, inner_rhs.take(), inner_lhs.take(), lhs.take()),
                                ),
                                (ref cmp, Multiply, Left) if cmp.is_comparison() => {
                                    // x cmp (y * z) <=> (x / z) cmp y
                                    use self::RangeClassification::*;
                                    match classify(inner_rhs) {
                                        Positive => Some(left_associate2(Divide, **cmp, lhs.take(), inner_rhs.take(), inner_lhs.take())),
                                        Negative => Some(left_associate2(Divide, flip_comp(*cmp), lhs.take(), inner_rhs.take(), inner_lhs.take())),
                                        Zero => Some(BinOp {
                                            kind: **cmp,
                                            left: lhs.take(),
                                            right: inner_rhs.take(),
                                        }),
                                        Unknown => None,
                                    }
                                }
                                (ref cmp, Multiply, Right) if cmp.is_comparison() => {
                                    // x cmp (y * z) <=> (x / y) cmp z
                                    use self::RangeClassification::*;
                                    match classify(inner_lhs) {
                                        Positive => Some(left_associate2(Divide, **cmp, lhs.take(), inner_lhs.take(), inner_rhs.take())),
                                        Negative => Some(left_associate2(Divide, flip_comp(*cmp), lhs.take(), inner_lhs.take(), inner_rhs.take())),
                                        Zero => Some(BinOp {
                                            kind: **cmp,
                                            left: lhs.take(),
                                            right: inner_lhs.take(),
                                        }),
                                        Unknown => None,
                                    }
                                }
                                (ref cmp, Divide, _) if cmp.is_comparison() => {
                                    // x cmp (y / z) <=> (x * z) cmp y
                                    // if y is constant, next pass will optimise this
                                    use self::RangeClassification::*;
                                    match classify(inner_rhs) {
                                        Positive => Some(left_associate2(Multiply, **cmp, lhs.take(), inner_rhs.take(), inner_lhs.take())),
                                        Negative => Some(left_associate2(Multiply, flip_comp(*cmp), lhs.take(), inner_rhs.take(), inner_lhs.take())),
                                        Zero => None, // division by 0!
                                        Unknown => None,
                                    }
                                }
                                _ => None,
                            }
                        }
                        UnaryOp {
                            kind: ref mut _inner_uop_kind,
                            value: ref mut _inner_expr,
                        } => {
                            // TODO
                            /* Most of these have inverses, but only on limited range
                             * Doing range analysis isn't cheap and might not be worth
                             * the effort.
                             */
                            None
                        }
                        Negate(ref mut inner_expr) => {
                            if binop_kind.is_comparison() {
                                // x > -y = -x < y and x < -y = -x > y equivalent for >= and <=
                                Some(BinOp {
                                    kind: flip_comp(binop_kind),
                                    left: Box::new(negate_expr(lhs.as_mut().take()).unwrap()),
                                    right: inner_expr.take(),
                                })
                            } else {
                                use ast::BinOpKind::*;
                                match binop_kind {
                                    Add => {
                                        // x + -y = x - y
                                        Some(BinOp {
                                            kind: Subtract,
                                            left: lhs.take(),
                                            right: inner_expr.take(),
                                        })
                                    }
                                    Subtract => {
                                        // x - -y = x + y
                                        Some(BinOp {
                                            kind: Add,
                                            left: lhs.take(),
                                            right: inner_expr.take(),
                                        })
                                    }
                                    Multiply | Divide => {
                                        // x * -y = -x * y and x/-y = -x/y
                                        Some(BinOp {
                                            kind: *binop_kind,
                                            left: Box::new(negate_expr(lhs.as_mut().take()).unwrap()),
                                            right: inner_expr.take(),
                                        })
                                    }
                                    _ => None,
                                }
                            }
                        }
                        _ => None,
                    }
                }
                (false, true) => {
                    match lhs.as_mut().kind {
                        BinOp {
                            kind: ref mut inner_binop_kind,
                            left: ref mut inner_lhs,
                            right: ref mut inner_rhs,
                        } => {
                            use self::ExprCmp::*;
                            use ast::BinOpKind::*;
                            let should_isolate = faster_change(inner_lhs, inner_rhs);
                            match (inner_binop_kind, binop_kind, should_isolate) {
                                (Add, Add, Left) => Some(right_associate(
                                    // (x + y) + z = x + (y + z)
                                    Add,
                                    inner_lhs.take(),
                                    inner_rhs.take(),
                                    rhs.take(),
                                )),
                                (Add, Add, Right) => Some(right_associate(
                                    // (x + y) + z = y + (x + z)
                                    Add,
                                    inner_rhs.take(),
                                    inner_lhs.take(),
                                    rhs.take(),
                                )),
                                (Add, Subtract, Left) => Some(right_associate2(
                                    // (x + y) - z = x + (y - z)
                                    Add,
                                    Subtract,
                                    inner_lhs.take(),
                                    inner_rhs.take(),
                                    rhs.take(),
                                )),
                                (Add, Subtract, Right) => Some(right_associate2(
                                    // (x + y) - z = y + (x - z)
                                    Add,
                                    Subtract,
                                    inner_rhs.take(),
                                    inner_lhs.take(),
                                    rhs.take(),
                                )),
                                //(Add, Multiply, _) => None, // (x + y) * z = x * z + y * z
                                //(Add, Divide, _) => None, // (x + y) / z = x/z + y/z
                                //(Add, Modulo, _) => None, // (x + y) % z
                                (Add, ref cmp, Left) if cmp.is_comparison() => Some(
                                    // (x + y) cmp z <=> x cmp (z - y)
                                    right_associate2(**cmp, Subtract, inner_lhs.take(), rhs.take(), inner_rhs.take()),
                                ),
                                (Add, ref cmp, Right) if cmp.is_comparison() => Some(
                                    // (x + y) cmp z <=> y cmp (z - x)
                                    right_associate2(**cmp, Subtract, inner_rhs.take(), rhs.take(), inner_lhs.take()),
                                ),
                                (Subtract, Add, Left) => Some(
                                    // (x - y) + z = x + (z - y)
                                    right_associate2(Add, Subtract, inner_lhs.take(), rhs.take(), inner_rhs.take()),
                                ),
                                (Subtract, Add, Right) => Some(
                                    // (x - y) + z = (x + z) - y
                                    left_associate2(Add, Subtract, inner_lhs.take(), rhs.take(), inner_rhs.take()),
                                ),
                                (Subtract, Subtract, Left) => Some(
                                    // (x - y) - z = x - (y + z)
                                    right_associate2(Subtract, Add, inner_lhs.take(), inner_rhs.take(), rhs.take()),
                                ),
                                (Subtract, Subtract, Right) => Some(
                                    // (x - y) - z = (x - z) - y
                                    left_associate2(Subtract, Subtract, inner_lhs.take(), rhs.take(), inner_rhs.take()),
                                ),
                                //(Subtract, Multiply, _) => None, // (x - y) * z = x * z - y * z
                                //(Subtract, Divide, _) => None, // (x - y) / z = x/z - y/z
                                //(Subtract, Modulo, _) => None, // (x - y) % z
                                (Subtract, ref cmp, Left) if cmp.is_comparison() => Some(
                                    // (x - y) cmp z <=> x cmp (y + z)
                                    right_associate2(**cmp, Add, inner_lhs.take(), inner_rhs.take(), rhs.take()),
                                ),
                                (Subtract, ref cmp, Right) if cmp.is_comparison() => Some(
                                    // (x - y) cmp z <=> y flip(cmp) (x - z)
                                    right_associate2(flip_comp(*cmp), Subtract, inner_rhs.take(), inner_lhs.take(), rhs.take()),
                                ),
                                //(Multiply, Add, _) => None, // (x * y) + z = x * (y + z/x)
                                //(Multiply, Subtract, _) => None, // (x * y) - z = x * (y - z/x)
                                (Multiply, Multiply, Left) => Some(
                                    // (x * y) * z = x * (y * z)
                                    right_associate(Multiply, inner_lhs.take(), inner_rhs.take(), rhs.take()),
                                ),
                                (Multiply, Multiply, Right) => Some(
                                    // (x * y) * z = y * (x * z)
                                    right_associate(Multiply, inner_rhs.take(), inner_lhs.take(), rhs.take()),
                                ),
                                (Multiply, Divide, Left) => Some(
                                    // (x * y) / z = x * (y / z)
                                    right_associate2(Multiply, Divide, inner_lhs.take(), inner_rhs.take(), rhs.take()),
                                ),
                                (Multiply, Divide, Right) => Some(
                                    // (x * y) / z = y * (x / z)
                                    right_associate2(Multiply, Divide, inner_rhs.take(), inner_lhs.take(), rhs.take()),
                                ),
                                // (Multiply, Modulo, _) => None // (x * y) % z
                                (Multiply, ref cmp, Left) if cmp.is_comparison() => {
                                    // (x * y) cmp z <=> x cmp (z/y)
                                    match classify(inner_rhs) {
                                        RangeClassification::Positive => Some(right_associate2(**cmp, Divide, inner_lhs.take(), rhs.take(), inner_rhs.take())),
                                        RangeClassification::Negative => Some(right_associate2(flip_comp(*cmp), Divide, inner_lhs.take(), rhs.take(), inner_rhs.take())),
                                        RangeClassification::Zero => Some(BinOp {
                                            kind: **cmp,
                                            left: inner_rhs.take(),
                                            right: rhs.take(),
                                        }),
                                        RangeClassification::Unknown => None,
                                    }
                                }
                                (Multiply, ref cmp, Right) if cmp.is_comparison() => {
                                    // (x * y) cmp z <=> y cmp (z/x)
                                    match classify(inner_lhs) {
                                        RangeClassification::Positive => Some(right_associate2(**cmp, Divide, inner_rhs.take(), rhs.take(), inner_lhs.take())),
                                        RangeClassification::Negative => Some(right_associate2(flip_comp(*cmp), Divide, inner_rhs.take(), rhs.take(), inner_lhs.take())),
                                        RangeClassification::Zero => Some(BinOp {
                                            kind: **cmp,
                                            left: inner_lhs.take(),
                                            right: rhs.take(),
                                        }),
                                        RangeClassification::Unknown => None,
                                    }
                                }
                                //(Divide, Add, _) => None, // (x / y) + z = (x + yz) / y
                                //(Divide, Subtract, _) => None, // (x / y) - z = (x - yz) / y
                                (Divide, Multiply, Left) => Some(
                                    // (x / y) * z = x * (z / y)
                                    right_associate2(Multiply, Divide, inner_lhs.take(), rhs.take(), inner_rhs.take()),
                                ),
                                (Divide, Multiply, Right) => Some(
                                    // (x / y) * z = (x * z) / y
                                    left_associate2(Multiply, Divide, inner_lhs.take(), rhs.take(), inner_rhs.take()),
                                ),
                                (Divide, Divide, Left) => Some(
                                    // (x / y) / z = x / (y *  z)
                                    right_associate2(Divide, Multiply, inner_lhs.take(), inner_rhs.take(), rhs.take()),
                                ),
                                (Divide, Divide, Right) => Some(
                                    // (x / y) / z = (x / z) / y
                                    left_associate(Divide, inner_lhs.take(), rhs.take(), inner_rhs.take()),
                                ),
                                // (Divide, Modulo, _) => None // (x / y) % z
                                (Divide, ref cmp, _) if cmp.is_comparison() => {
                                    // (x / y) cmp z <=> x cmp (z * y)
                                    // if y is constant, leave the (x / z) cmp y optimisation to the next pass
                                    match classify(inner_rhs) {
                                        RangeClassification::Positive => Some(right_associate2(**cmp, Multiply, inner_lhs.take(), rhs.take(), inner_rhs.take())),
                                        RangeClassification::Negative => Some(right_associate2(flip_comp(*cmp), Multiply, inner_lhs.take(), rhs.take(), inner_rhs.take())),
                                        RangeClassification::Zero => None, // could rewrite like in multiply, but it would change the semantics (i.e. throwing exception vs no exception)
                                        RangeClassification::Unknown => None,
                                    }
                                }
                                _ => None,
                            }
                        }
                        UnaryOp {
                            kind: ref mut _inner_uop_kind,
                            value: ref mut _inner_expr,
                        } => {
                            // TODO
                            /* Most of these have inverses, but only on limited range
                             * Doing range analysis isn't cheap and might not be worth
                             * the effort.
                             */
                            None
                        }
                        Negate(ref mut inner_expr) => {
                            if binop_kind.is_comparison() {
                                // -x > y = x < -y and -x < y = x > -y equivalent for >= and <=
                                let new = BinOp {
                                    kind: flip_comp(binop_kind),
                                    left: inner_expr.take(),
                                    right: Box::new(negate_expr(rhs.as_mut().take()).unwrap()),
                                };
                                Some(new)
                            } else {
                                use ast::BinOpKind::*;
                                match binop_kind {
                                    Add => {
                                        // -x + y = y - x
                                        let new = BinOp {
                                            kind: Subtract,
                                            left: rhs.take(),
                                            right: inner_expr.take(),
                                        };
                                        Some(new)
                                    }
                                    Subtract => {
                                        // -x - y = -y - x
                                        let new = BinOp {
                                            kind: Subtract,
                                            left: Box::new(negate_expr(rhs.as_mut().take()).unwrap()),
                                            right: inner_expr.take(),
                                        };
                                        Some(new)
                                    }
                                    Multiply | Divide => {
                                        // -x * y = x * -y and -x/y = x/-y
                                        let new = BinOp {
                                            kind: *binop_kind,
                                            left: inner_expr.take(),
                                            right: Box::new(negate_expr(rhs.as_mut().take()).unwrap()),
                                        };
                                        Some(new)
                                    }
                                    Pow if is_even(rhs) => {
                                        // (-x)^y = x^y if y is even
                                        let new = BinOp {
                                            kind: Pow,
                                            left: inner_expr.take(),
                                            right: rhs.take(),
                                        };
                                        Some(new)
                                    }
                                    _ => None,
                                }
                            }
                        }
                        _ => None,
                    }
                }
            }
        }
        _ => {
            None
        }
    }
}

enum ExprCmp {
    Left,
    Right,
    Incomparable,
}

fn faster_change(left: &Expr, right: &Expr) -> ExprCmp {
    // TODO potentially analyse iteration depth
    match (is_constant(left), is_constant(right)) {
        (true, true) | (false, false) => ExprCmp::Incomparable,
        (true, false) => ExprCmp::Right,
        (false, true) => ExprCmp::Left,
    }
}

fn is_constant(e: &Expr) -> bool {
    match e.kind {
        Literal(_) => true,
        Ident(_) => false, // TODO might actually be true, but need to know the expression that produces that
        Negate(_) | BinOp { .. } | UnaryOp { .. } | Cast { .. } => e.children().all(is_constant),
        If {
            cond: ref _c,
            on_true: ref te,
            on_false: ref fe,
        } => is_constant(te) && is_constant(fe), // should push down the calc into the branches then
        _ => false,
    }
}

/// Return true if e is definitely an even number.
/// May return false for expressions that are actually even, if they cannot guaranteed to be so.
fn is_even(e: &Expr) -> bool {
    match e.kind {
        Literal(ref kind) => {
            use ast::LiteralKind::*;
            match kind {
                BoolLiteral(_) => false,
                I8Literal(n) => n.is_even(),
                I16Literal(n) => n.is_even(),
                I32Literal(n) => n.is_even(),
                I64Literal(n) => n.is_even(),
                U8Literal(n) => n.is_even(),
                U16Literal(n) => n.is_even(),
                U32Literal(n) => n.is_even(),
                U64Literal(n) => n.is_even(),
                F32Literal(bits) => {
                    let n = f32::from_bits(*bits);
                    n % 2.0 == 0.0
                }
                F64Literal(bits) => {
                    let n = f64::from_bits(*bits);
                    n % 2.0 == 0.0
                }
                StringLiteral(_) => false,
            }
        }
        Ident(_) => false, // TODO might actually be true, but need to know the expression that produces that
        Negate(ref e) => is_even(e),
        BinOp {
            kind: ref k,
            left: ref l,
            right: ref r,
        } => {
            use ast::BinOpKind::*;
            match k {
                Add | Subtract => is_even(l) && is_even(r), // TODO || (is_odd(l) && is_odd(r))
                Min | Max => is_even(l) && is_even(r),
                Multiply => is_even(l) || is_even(r),
                Pow => is_even(l),
                _ => false,
            }
        }
        UnaryOp { .. } => false,
        Cast { kind: ref k, child_expr: ref e } => k.is_numeric() && is_even(e),
        If {
            cond: ref _c,
            on_true: ref te,
            on_false: ref fe,
        } => is_even(te) && is_even(fe),
        _ => false,
    }
}

fn flip_comp(k: &BinOpKind) -> BinOpKind {
    use ast::BinOpKind::*;
    match k {
        Equal | NotEqual => k.clone(), // nothing to flip here
        LessThan => GreaterThan,
        GreaterThan => LessThan,
        LessThanOrEqual => GreaterThanOrEqual,
        GreaterThanOrEqual => LessThanOrEqual,
        _ => panic!("Can't flip direction of {:?}", k),
    }
}

fn flip_logical(k: &BinOpKind) -> BinOpKind {
    use ast::BinOpKind::*;
    match k {
        LogicalAnd => LogicalOr,
        LogicalOr => LogicalAnd,
        _ => panic!("Can't flip logic of {:?}", k),
    }
}

fn right_associate(op: BinOpKind, e1: Box<Expr>, e2: Box<Expr>, e3: Box<Expr>) -> ExprKind {
    let ty = e1.ty.clone();
    BinOp {
        kind: op,
        left: e1,
        right: Box::new(new_expr(BinOp { kind: op, left: e2, right: e3 }, ty).unwrap()),
    }
}

fn right_associate2(op1: BinOpKind, op2: BinOpKind, e1: Box<Expr>, e2: Box<Expr>, e3: Box<Expr>) -> ExprKind {
    let ty = e1.ty.clone();
    BinOp {
        kind: op1,
        left: e1,
        right: Box::new(new_expr(BinOp { kind: op2, left: e2, right: e3 }, ty).unwrap()),
    }
}

fn left_associate(op: BinOpKind, e1: Box<Expr>, e2: Box<Expr>, e3: Box<Expr>) -> ExprKind {
    let ty = e1.ty.clone();
    BinOp {
        kind: op,
        left: Box::new(new_expr(BinOp { kind: op, left: e1, right: e2 }, ty).unwrap()),
        right: e3,
    }
}

fn left_associate2(op1: BinOpKind, op2: BinOpKind, e1: Box<Expr>, e2: Box<Expr>, e3: Box<Expr>) -> ExprKind {
    let ty = e1.ty.clone();
    BinOp {
        kind: op2,
        left: Box::new(new_expr(BinOp { kind: op1, left: e1, right: e2 }, ty).unwrap()),
        right: e3,
    }
}

#[derive(PartialEq, Eq, Debug)]
pub enum RangeClassification {
    Positive,
    Negative,
    Zero,
    Unknown,
}

impl RangeClassification {
    fn negate(&self) -> RangeClassification {
        match self {
            RangeClassification::Positive => RangeClassification::Negative,
            RangeClassification::Negative => RangeClassification::Positive,
            RangeClassification::Zero => RangeClassification::Zero,
            RangeClassification::Unknown => RangeClassification::Unknown,
        }
    }
}

fn classify(e: &Expr) -> RangeClassification {
    match e.kind {
        Literal(ref lkind) => match lkind {
            BoolLiteral(_) => RangeClassification::Unknown,
            I8Literal(n) => {
                if *n == 0 {
                    RangeClassification::Zero
                } else if *n < 0 {
                    RangeClassification::Negative
                } else {
                    RangeClassification::Positive
                }
            }
            I16Literal(n) => {
                if *n == 0 {
                    RangeClassification::Zero
                } else if *n < 0 {
                    RangeClassification::Negative
                } else {
                    RangeClassification::Positive
                }
            }
            I32Literal(n) => {
                if *n == 0 {
                    RangeClassification::Zero
                } else if *n < 0 {
                    RangeClassification::Negative
                } else {
                    RangeClassification::Positive
                }
            }
            I64Literal(n) => {
                if *n == 0 {
                    RangeClassification::Zero
                } else if *n < 0 {
                    RangeClassification::Negative
                } else {
                    RangeClassification::Positive
                }
            }
            U8Literal(n) => {
                if *n == 0 {
                    RangeClassification::Zero
                } else {
                    RangeClassification::Positive
                }
            }
            U16Literal(n) => {
                if *n == 0 {
                    RangeClassification::Zero
                } else {
                    RangeClassification::Positive
                }
            }
            U32Literal(n) => {
                if *n == 0 {
                    RangeClassification::Zero
                } else {
                    RangeClassification::Positive
                }
            }
            U64Literal(n) => {
                if *n == 0 {
                    RangeClassification::Zero
                } else {
                    RangeClassification::Positive
                }
            }
            F32Literal(bits) => {
                let n = f32::from_bits(*bits);
                if n.is_nan() {
                    RangeClassification::Unknown
                } else if n == 0.0 {
                    RangeClassification::Zero
                } else if n < 0.0 {
                    RangeClassification::Negative
                } else {
                    RangeClassification::Positive
                }
            }
            F64Literal(bits) => {
                let n = f64::from_bits(*bits);
                if n.is_nan() {
                    RangeClassification::Unknown
                } else if n == 0.0 {
                    RangeClassification::Zero
                } else if n < 0.0 {
                    RangeClassification::Negative
                } else {
                    RangeClassification::Positive
                }
            }
            StringLiteral(_) => RangeClassification::Unknown,
        },
        Ident(_) => RangeClassification::Unknown, // TODO need to know the expression that produces that
        Negate(ref e) => classify(e).negate(),
        BinOp {
            kind: ref k,
            left: ref l,
            right: ref r,
        } => {
            use self::RangeClassification::*;
            use ast::BinOpKind::*;

            let lc = classify(l);
            let rc = classify(r);
            match k {
                Add => match (lc, rc) {
                    (Positive, Positive) => Positive,
                    (Positive, Negative) | (Negative, Positive) => Unknown,
                    (Negative, Negative) => Negative,
                    (Zero, Zero) => Zero,
                    (Positive, Zero) | (Zero, Positive) => Positive,
                    (Negative, Zero) | (Zero, Negative) => Negative,
                    _ => Unknown,
                },
                Subtract => match (lc, rc) {
                    (Positive, Positive) => Unknown,
                    (Positive, Negative) => Positive,
                    (Negative, Positive) => Negative,
                    (Negative, Negative) => Unknown,
                    (Zero, Zero) => Zero,
                    (Positive, Zero) => Positive,
                    (Zero, Positive) => Negative,
                    (Negative, Zero) => Negative,
                    (Zero, Negative) => Positive,
                    _ => Unknown,
                },
                Multiply => match (lc, rc) {
                    (Positive, Positive) => Positive,
                    (Positive, Negative) | (Negative, Positive) => Negative,
                    (Negative, Negative) => Positive,
                    (Zero, _) | (_, Zero) => Zero,
                    _ => Unknown,
                },
                Divide => {
                    match (lc, rc) {
                        (Positive, Positive) => Positive,
                        (Positive, Negative) | (Negative, Positive) => Negative,
                        (Negative, Negative) => Positive,
                        (Zero, Zero) | (Negative, Zero) | (Positive, Zero) => Unknown, // division by 0!
                        (Zero, Positive) | (Zero, Negative) => Zero,
                        _ => Unknown,
                    }
                }
                Modulo => Unknown, // TODO check how this behaves
                Max => match (lc, rc) {
                    (Positive, Positive) => Positive,
                    (Positive, Negative) | (Negative, Positive) => Positive,
                    (Negative, Negative) => Negative,
                    (Zero, Zero) | (Negative, Zero) | (Zero, Negative) => Zero,
                    (Zero, Positive) | (Positive, Zero) => Positive,
                    _ => Unknown,
                },
                Min => match (lc, rc) {
                    (Positive, Positive) => Positive,
                    (Positive, Negative) | (Negative, Positive) => Negative,
                    (Negative, Negative) => Negative,
                    (Negative, Zero) | (Zero, Negative) => Negative,
                    (Zero, Zero) | (Zero, Positive) | (Positive, Zero) => Zero,
                    _ => Unknown,
                },
                Pow => {
                    match (lc, rc) {
                        (Positive, Positive) | (Positive, Negative) => Positive,
                        (Negative, Positive) | (Negative, Negative) => Unknown, // TODO could check with is_even
                        (_, Zero) => Positive,                                  // x^0 = 1
                        (Zero, Positive) => Zero,
                        (Zero, Negative) => Unknown, // division by 0!
                        _ => Unknown,
                    }
                }
                _ => Unknown,
            }
        }
        UnaryOp { kind: ref k, value: ref e } => {
            use self::RangeClassification::*;
            use ast::UnaryOpKind::*;
            let ec = classify(e);
            match k {
                Exp => Positive, // e^x > 0
                Log => {
                    match ec {
                        Zero => Negative, // -inf
                        _ => Unknown,
                    }
                }
                Sqrt => {
                    match ec {
                        Positive => Positive,
                        Negative => Unknown, // complex
                        Zero => Zero,
                        Unknown => Unknown,
                    }
                }
                Sin => match ec {
                    Zero => Zero,
                    _ => Unknown,
                },
                Cos => match ec {
                    Zero => Positive,
                    _ => Unknown,
                },
                Tan => match ec {
                    Zero => Zero,
                    _ => Unknown,
                },
                Erf => match ec {
                    Positive => Positive,
                    Negative => Negative,
                    Zero => Zero,
                    Unknown => Unknown,
                },
                _ => Unknown,
            }
        }
        Cast { kind: ref k, child_expr: ref e } => {
            use self::RangeClassification::*;
            if k.is_numeric() {
                if let Scalar(ty) = e.ty {
                    if k.is_unsigned_integer() {
                        let ec = classify(e);
                        match ec {
                            Positive | Negative => Positive,
                            Zero => Zero,
                            Unknown => Unknown,
                        }
                    } else if ty.is_unsigned_integer() {
                        if ty.is_strict_upcast(k) && k.is_integer() {
                            let ec = classify(e);
                            match ec {
                                Positive => Positive,
                                Zero => Zero,
                                Negative | Unknown => Unknown, // u* can't be negative
                            }
                        } else {
                            Unknown // might overflow and behave weird
                        }
                    } else if k.is_float() && ty.is_float() {
                        classify(e) // should preserve signs and stuff
                    } else {
                        Unknown
                    }
                } else {
                    Unknown
                }
            } else {
                Unknown
            }
        }
        If {
            cond: ref _c,
            on_true: ref te,
            on_false: ref fe,
        } => {
            let tec = classify(te);
            let fec = classify(fe);
            if tec == fec {
                tec
            } else {
                RangeClassification::Unknown
            }
        }
        _ => RangeClassification::Unknown,
    }
}
