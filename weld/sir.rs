//! Sequential IR for Weld programs

use std::fmt;
use std::collections::{BTreeMap, HashMap};

use super::ast::*;
use super::ast::ExprKind::*;
use super::error::*;
use super::pretty_print::*;
use super::util::SymbolGenerator;

type BasicBlockId = usize;

/// A non-terminating statement inside a basic block.
pub enum Statement {
    /// output, op, type, left, right
    AssignBinOp(Symbol, BinOpKind, Type, Symbol, Symbol),
    /// output, value
    Assign(Symbol, Symbol),
    /// output, value
    AssignLiteral(Symbol, LiteralKind),
    /// output, builder, value
    DoMerge(Symbol, Symbol, Symbol),
    /// output, builder
    GetResult(Symbol, Symbol),
    /// output, builder type
    CreateBuilder(Symbol, Type),
}

/// A terminating statement inside a basic block.
pub enum Terminator {
    /// condition, on_true, on_false
    Branch(Symbol, BasicBlockId, BasicBlockId),
    Jump(BasicBlockId),
    Return(Symbol),
    ParallelFor {
        data: Symbol,
        builder: Symbol,
        body: BasicBlockId,
        data_arg: Symbol,
        builder_arg: Symbol,
        exit: BasicBlockId,
        result: Symbol,
    },
    Crash
}

/// A basic block inside a SIR program
pub struct BasicBlock {
    id: BasicBlockId,
    statements: Vec<Statement>,
    terminator: Terminator
}

pub struct SirFunction {
    params: Vec<TypedParameter>,
    locals: HashMap<Symbol, Type>,
    blocks: Vec<BasicBlock>,
    sym_gen: SymbolGenerator,
}

impl SirFunction {
    pub fn new() -> SirFunction {
        SirFunction {
            params: vec![],
            blocks: vec![],
            locals: HashMap::new(),
            sym_gen: SymbolGenerator::new(),
        }
    }

    /// Add a new basic block and return its block ID.
    pub fn add_block(&mut self) -> BasicBlockId {
        let block = BasicBlock {
            id: self.blocks.len(),
            statements: vec![],
            terminator: Terminator::Crash
        };
        self.blocks.push(block);
        self.blocks.len() - 1
    }

    /// Add a local variable of the given type and return a symbol for it.
    pub fn add_local(&mut self, ty: &Type) -> Symbol {
        let sym = self.sym_gen.new_symbol("tmp");
        self.locals.insert(sym.clone(), ty.clone());
        sym
    }
}

impl BasicBlock {
    pub fn add_statement(&mut self, statement: Statement) {
        self.statements.push(statement);
    }
}

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::Statement::*;
        match *self {
            AssignBinOp(ref out, ref op, ref ty, ref left, ref right) => {
                write!(f, "{} = {} {} {} {}", out, op, print_type(ty), left, right)
            },
            Assign(ref out, ref value) => write!(f, "{} = {}", out, value),
            AssignLiteral(ref out, ref lit) => write!(f, "{} = {}", out, print_literal(lit)),
            DoMerge(ref out, ref bld, ref elem) => write!(f, "{} = merge {} {}", out, bld, elem),
            GetResult(ref out, ref value) => write!(f, "{} = result {}", out, value),
            CreateBuilder(ref out, ref ty) => write!(f, "{} = new {}", out, print_type(ty)),
        }
    }
}

impl fmt::Display for Terminator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::Terminator::*;
        match *self {
            Branch(ref cond, ref on_true, ref on_false) => {
                write!(f, "branch {} B{} B{}", cond, on_true, on_false)
            },
            ParallelFor {
                ref data, ref builder, body, ref data_arg, ref builder_arg, exit, ref result
            } => {
                write!(f, "for {} {} B{} {} {} B{} {}",
                    data, builder, body, data_arg, builder_arg, exit, result)
            },
            Jump(block) => write!(f, "jump B{}", block),
            Return(ref sym) => write!(f, "return {}", sym),
            Crash => write!(f, "crash")
        }
    }
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "B{}:\n", self.id)?;
        for stmt in &self.statements {
            write!(f, "  {}\n", stmt)?;
        }
        write!(f, "  {}\n", self.terminator)?;
        Ok(())
    }
}

impl fmt::Display for SirFunction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Params:\n")?;
        for param in &self.params {
            write!(f, "  {}: {}\n", param.name, print_type(&param.ty))?;
        }
        write!(f, "Locals:\n")?;
        let locals_sorted: BTreeMap<&Symbol, &Type> = self.locals.iter().collect();
        for (name, ty) in locals_sorted {
            write!(f, "  {}: {}\n", name, print_type(ty))?;
        }
        for block in &self.blocks {
            write!(f, "{}", block)?;
        }
        Ok(())
    }
}

pub fn ast_to_sir(expr: &TypedExpr) -> WeldResult<SirFunction> {
    if let Lambda(ref params, ref body) = expr.kind {
        let mut func = SirFunction::new();
        func.sym_gen = SymbolGenerator::from_expression(expr);
        if func.sym_gen.next_id("tmp") == 0 {
            func.sym_gen.new_symbol("tmp");
        }
        func.params = params.clone();
        let first_block = func.add_block();
        let (res_block, res_sym) = gen_expr(body, &mut func, first_block)?;
        func.blocks[res_block].terminator = Terminator::Return(res_sym);
        Ok((func))
    } else {
        weld_err!("Expression passed to ast_to_sir was not a Lambda")
    }
}

/// Generate code to compute the expression `expr` starting at the current tail of `cur_block`,
/// possibly creating new basic blocks in the process. Return the basic block that the expression
/// will be ready in, and its symbol therein.
fn gen_expr(
    expr: &TypedExpr,
    func: &mut SirFunction,
    cur_block: BasicBlockId
) -> WeldResult<(BasicBlockId, Symbol)> {
    use self::Statement::*;
    use self::Terminator::*;
    match expr.kind {
        Ident(ref sym) => Ok((cur_block, sym.clone())),

        Literal(lit) => {
            let res_sym = func.add_local(&expr.ty);
            func.blocks[cur_block].add_statement(AssignLiteral(res_sym.clone(), lit));
            Ok((cur_block, res_sym))
        },

        BinOp(kind, ref left, ref right) => {
            let (cur_block, left_sym) = gen_expr(left, func, cur_block)?;
            let (cur_block, right_sym) = gen_expr(right, func, cur_block)?;
            let res_sym = func.add_local(&expr.ty);
            func.blocks[cur_block].add_statement(
                AssignBinOp(res_sym.clone(), kind, left.ty.clone(), left_sym, right_sym));
            Ok((cur_block, res_sym))
        },

        If(ref cond, ref on_true, ref on_false) => {
            let (cur_block, cond_sym) = gen_expr(cond, func, cur_block)?;
            let true_block = func.add_block();
            let false_block = func.add_block();
            func.blocks[cur_block].terminator = Branch(cond_sym, true_block, false_block);
            let (true_block, true_sym) = gen_expr(on_true, func, true_block)?;
            let (false_block, false_sym) = gen_expr(on_false, func, false_block)?;
            let res_sym = func.add_local(&expr.ty);
            let res_block = func.add_block();
            func.blocks[true_block].add_statement(Assign(res_sym.clone(), true_sym));
            func.blocks[true_block].terminator = Jump(res_block);
            func.blocks[false_block].add_statement(Assign(res_sym.clone(), false_sym));
            func.blocks[false_block].terminator = Jump(res_block);
            Ok((res_block, res_sym))
        },

        Merge(ref builder, ref elem) => {
            let (cur_block, builder_sym) = gen_expr(builder, func, cur_block)?;
            let (cur_block, elem_sym) = gen_expr(elem, func, cur_block)?;
            let res_sym = func.add_local(&expr.ty);
            func.blocks[cur_block].add_statement(DoMerge(res_sym.clone(), builder_sym, elem_sym));
            Ok((cur_block, res_sym))
        },

        Res(ref builder) => {
            let (cur_block, builder_sym) = gen_expr(builder, func, cur_block)?;
            let res_sym = func.add_local(&expr.ty);
            func.blocks[cur_block].add_statement(GetResult(res_sym.clone(), builder_sym));
            Ok((cur_block, res_sym))
        },

        NewBuilder => {
            let res_sym = func.add_local(&expr.ty);
            func.blocks[cur_block].add_statement(CreateBuilder(res_sym.clone(), expr.ty.clone()));
            Ok((cur_block, res_sym))
        },

        For(ref data, ref builder, ref update) => {
            if let Lambda(ref params, ref body) = update.kind {
                let (cur_block, data_sym) = gen_expr(data, func, cur_block)?;
                let (cur_block, builder_sym) = gen_expr(builder, func, cur_block)?;
                let body_block = func.add_block();
                let (body_end_block, body_res) = gen_expr(body, func, body_block)?;
                func.blocks[body_end_block].terminator = Return(body_res);
                let exit_block = func.add_block();
                let res_sym = func.add_local(&expr.ty);
                func.blocks[cur_block].terminator = ParallelFor {
                    data: data_sym,
                    builder: builder_sym,
                    body: body_block,
                    data_arg: params[0].name.clone(),
                    builder_arg: params[1].name.clone(),
                    exit: exit_block,
                    result: res_sym.clone()
                };
                Ok((exit_block, res_sym))
            } else {
                weld_err!("Argument to For was not a Lambda: {}", print_expr(update))
            }
        },

        _ => weld_err!("Unsupported expression: {}", print_expr(expr))
    }
}
