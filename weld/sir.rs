//! Sequential IR for Weld programs

use std::fmt;
use std::collections::{BTreeMap, HashMap, HashSet};

use super::ast::*;
use super::ast::ExprKind::*;
use super::error::*;
use super::pretty_print::*;
use super::util::SymbolGenerator;

pub type BasicBlockId = usize;
pub type FunctionId = usize;

/// A non-terminating statement inside a basic block.
#[derive(Clone)]
pub enum Statement {
    AssignBinOp {
        output: Symbol,
        op: BinOpKind,
        ty: Type,
        left: Symbol,
        right: Symbol
    },
    CastOp {
        output: Symbol,
        old_ty: Type,
        new_ty: Type,
        child: Symbol,
    },
    Assign {
        output: Symbol,
        value: Symbol
    },
    AssignLiteral {
        output: Symbol,
        value: LiteralKind
    },
    DoMerge {
        builder: Symbol,
        value: Symbol
    },
    GetResult {
        output: Symbol,
        builder: Symbol
    },
    CreateBuilder {
        output: Symbol,
        ty: Type
    },
}

#[derive(Clone)]
pub struct ParallelForData {
    pub data: Symbol,
    pub builder: Symbol,
    pub data_arg: Symbol,
    pub builder_arg: Symbol,
    pub body: FunctionId,
    pub cont: FunctionId
}

/// A terminating statement inside a basic block.
#[derive(Clone)]
pub enum Terminator {
    Branch {
        cond: Symbol,
        on_true: BasicBlockId,
        on_false: BasicBlockId
    },
    JumpBlock(BasicBlockId),
    JumpFunction(FunctionId),
    ProgramReturn(Symbol),
    EndFunction,
    ParallelFor(ParallelForData),
    Crash
}

/// A basic block inside a SIR program
#[derive(Clone)]
pub struct BasicBlock {
    pub id: BasicBlockId,
    pub statements: Vec<Statement>,
    pub terminator: Terminator
}

pub struct SirFunction {
    pub id: FunctionId,
    pub params: HashMap<Symbol, Type>,
    pub locals: HashMap<Symbol, Type>,
    pub blocks: Vec<BasicBlock>
}

pub struct SirProgram {
    /// funcs[0] is the main function
    pub funcs: Vec<SirFunction>,
    pub ret_ty: Type,
    pub top_params: Vec<TypedParameter>,
    sym_gen: SymbolGenerator
}

impl SirProgram {
    pub fn new(ret_ty: &Type, top_params: &Vec<TypedParameter>) -> SirProgram {
        let mut prog = SirProgram {
            funcs: vec![],
            ret_ty: ret_ty.clone(),
            top_params: top_params.clone(),
            sym_gen: SymbolGenerator::new()
        };
        /// add main
        prog.add_func();
        prog
    }

    pub fn add_func(&mut self) -> FunctionId {
        let func = SirFunction {
            id: self.funcs.len(),
            params: HashMap::new(),
            blocks: vec![],
            locals: HashMap::new()
        };
        self.funcs.push(func);
        self.funcs.len() - 1       
    }

    /// Add a local variable of the given type and return a symbol for it.
    pub fn add_local(&mut self, ty: &Type, func: FunctionId) -> Symbol {
        let sym = self.sym_gen.new_symbol(format!("fn{}_tmp", func).as_str());
        self.funcs[func].locals.insert(sym.clone(), ty.clone());
        sym
    }

    /// Add a local variable of the given type and name
    pub fn add_local_named(&mut self, ty: &Type, sym: &Symbol, func: FunctionId) {
        self.funcs[func].locals.insert(sym.clone(), ty.clone());
    }
}

impl SirFunction {
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
            AssignBinOp { ref output, ref op, ref ty, ref left, ref right } => {
                write!(f, "{} = {} {} {} {}", output, op, print_type(ty), left, right)
            },
            CastOp { ref output, ref new_ty, ref child, .. } => {
                write!(f, "{} = cast({}, {})", output, child, print_type(new_ty))
            },
            Assign { ref output, ref value } => write!(f, "{} = {}", output, value),
            AssignLiteral { ref output, ref value } => write!(f, "{} = {}", output,
                print_literal(value)),
            DoMerge { ref builder, ref value } => write!(f, "merge {} {}", builder, value),
            GetResult { ref output, ref builder } => write!(f, "{} = result {}", output, builder),
            CreateBuilder { ref output, ref ty }  => write!(f, "{} = new {}", output,
                print_type(ty)),
        }
    }
}

impl fmt::Display for Terminator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::Terminator::*;
        match *self {
            Branch { ref cond, ref on_true, ref on_false } => {
                write!(f, "branch {} B{} B{}", cond, on_true, on_false)
            },
            ParallelFor(ref pf) => {
                write!(f, "for {} {} {} {} F{} F{}",
                    pf.data, pf.builder, pf.data_arg, pf.builder_arg, pf.body, pf.cont)?;
                Ok(())
            },
            JumpBlock(block) => write!(f, "jump B{}", block),
            JumpFunction(func) => write!(f, "jump F{}", func),
            ProgramReturn(ref sym) => write!(f, "return {}", sym),
            EndFunction => write!(f, "end"),
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
        write!(f, "F{}:\n", self.id)?;
        write!(f, "Params:\n")?;
        let params_sorted: BTreeMap<&Symbol, &Type> = self.params.iter().collect();
        for (name, ty) in params_sorted {
            write!(f, "  {}: {}\n", name, print_type(ty))?;
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

impl fmt::Display for SirProgram {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for func in &self.funcs {
            write!(f, "{}\n", func)?;
        }
        Ok(())
    }
}

/// Recursive helper function for sir_param_correction. env contains the symbol to type mappings
/// that have been defined previously in the program. Any symbols that need to be passed in
/// as closure parameters to func_id will be added to closure (so that func_id's
/// callers can also add these symbols to their parameters list, if necessary).
fn sir_param_correction_helper(prog: &mut SirProgram, func_id: FunctionId,
env: &mut HashMap<Symbol, Type>, closure: &mut HashSet<Symbol>) {
    for (name, ty) in &prog.funcs[func_id].params {
        env.insert(name.clone(), ty.clone());
    }
    for (name, ty) in &prog.funcs[func_id].locals {
        env.insert(name.clone(), ty.clone());
    }
    // All symbols are unique, so there is no need to remove stuff from env at any point.
    for block in prog.funcs[func_id].blocks.clone() {
        let mut vars = vec![];
        for statement in &block.statements {
            use self::Statement::*;
            match *statement {
                AssignBinOp { ref left, ref right, .. } => {
                    vars.push(left.clone());
                    vars.push(right.clone());
                },
                CastOp { ref child, .. } => {
                    vars.push(child.clone());
                },
                Assign { ref value, .. } => vars.push(value.clone()),
                DoMerge { ref builder, ref value } => {
                    vars.push(builder.clone());
                    vars.push(value.clone());
                },
                GetResult { ref builder, .. } => vars.push(builder.clone()),
                _ => {}
            }   
        }
        for var in &vars {
            if prog.funcs[func_id].locals.get(&var) == None {
                prog.funcs[func_id].params.insert(var.clone(), env.get(&var).unwrap().clone());
                closure.insert(var.clone());
            }
        }
        let mut inner_closure = HashSet::new();
        use self::Terminator::*;
        match block.terminator {
            ParallelFor(ref pf) => {
                sir_param_correction_helper(prog, pf.body, env, &mut inner_closure);
                sir_param_correction_helper(prog, pf.cont, env, &mut inner_closure);
            },
            JumpFunction(jump_func) => {
                sir_param_correction_helper(prog, jump_func, env, &mut inner_closure);
            },
            _ => {}       
        }
        for var in inner_closure {
            if prog.funcs[func_id].locals.get(&var) == None {
                prog.funcs[func_id].params.insert(var.clone(), env.get(&var).unwrap().clone());
                closure.insert(var.clone());
            }
        }
    }
}

/// gen_expr may result in the use of symbols across function boundaries,
/// so ast_to_sir calls sir_param_correction to correct function parameters
/// to ensure that such symbols (the closure) are passed in as parameters.
fn sir_param_correction(prog: &mut SirProgram) -> WeldResult<()> {
    let mut env = HashMap::new();
    let mut closure = HashSet::new();
    sir_param_correction_helper(prog, 0, &mut env, &mut closure);
    let ref func = prog.funcs[0];
    for name in closure {
        if func.params.get(&name) == None {
            weld_err!("Unbound symbol {}#{}", name.name, name.id)?;
        }
    }
    Ok(())
}

/// Convert an AST to a SIR program. Symbols must be unique in expr.
pub fn ast_to_sir(expr: &TypedExpr) -> WeldResult<SirProgram> {
    if let Lambda { ref params, ref body } = expr.kind {
        let mut prog = SirProgram::new(&expr.ty, params);
        prog.sym_gen = SymbolGenerator::from_expression(expr);
        for tp in params {
            prog.funcs[0].params.insert(tp.name.clone(), tp.ty.clone());
        }
        let first_block = prog.funcs[0].add_block();
        let (res_func, res_block, res_sym) = gen_expr(body, &mut prog, 0, first_block)?;
        prog.funcs[res_func].blocks[res_block].terminator = Terminator::ProgramReturn(res_sym);
        sir_param_correction(&mut prog)?;
        Ok((prog))
    } else {
        weld_err!("Expression passed to ast_to_sir was not a Lambda")
    }
}

/// Generate code to compute the expression `expr` starting at the current tail of `cur_block`,
/// possibly creating new basic blocks and functions in the process. Return the function and
/// basic block that the expression will be ready in, and its symbol therein.
fn gen_expr(
    expr: &TypedExpr,
    prog: &mut SirProgram,
    cur_func: FunctionId,
    cur_block: BasicBlockId
) -> WeldResult<(FunctionId, BasicBlockId, Symbol)> {
    use self::Statement::*;
    use self::Terminator::*;
    match expr.kind {
        Ident(ref sym) => {
            Ok((cur_func, cur_block, sym.clone()))
        },

        Literal(lit) => {
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(
                AssignLiteral { output: res_sym.clone(), value: lit });
            Ok((cur_func, cur_block, res_sym))
        },

        Let { ref name, ref value, ref body } => {
            let (cur_func, cur_block, val_sym) = gen_expr(value, prog, cur_func, cur_block)?;
            prog.add_local_named(&value.ty, name, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(
                Assign { output: name.clone(), value: val_sym });
            let (cur_func, cur_block, res_sym) = gen_expr(body, prog, cur_func, cur_block)?;
            Ok((cur_func, cur_block, res_sym))
        },

        BinOp { kind, ref left, ref right } => {
            let (cur_func, cur_block, left_sym) = gen_expr(left, prog, cur_func, cur_block)?;
            let (cur_func, cur_block, right_sym) = gen_expr(right, prog, cur_func, cur_block)?;
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(
                AssignBinOp {
                    output: res_sym.clone(),
                    op: kind,
                    ty: left.ty.clone(),
                    left: left_sym,
                    right: right_sym
                });
            Ok((cur_func, cur_block, res_sym))
        },

        Cast { ref child_expr, .. } => {
            let (cur_func, cur_block, child_sym) = gen_expr(child_expr, prog, cur_func, cur_block)?;
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(
                CastOp {
                    output: res_sym.clone(),
                    old_ty: child_expr.ty.clone(),
                    new_ty: expr.ty.clone(),
                    child: child_sym,
                }
            );
            Ok((cur_func, cur_block, res_sym))
        },

        If { ref cond, ref on_true, ref on_false } => {
            let (cur_func, cur_block, cond_sym) = gen_expr(cond, prog, cur_func, cur_block)?;
            let true_block = prog.funcs[cur_func].add_block();
            let false_block = prog.funcs[cur_func].add_block();
            prog.funcs[cur_func].blocks[cur_block].terminator =
                Branch {
                    cond: cond_sym,
                    on_true: true_block,
                    on_false: false_block
                };
            let (true_func, true_block, true_sym) = gen_expr(on_true, prog, cur_func, true_block)?;
            let (false_func, false_block, false_sym) = gen_expr(on_false, prog, cur_func, false_block)?;
            let res_sym = prog.add_local(&expr.ty, true_func);
            prog.funcs[true_func].blocks[true_block].add_statement(
                Assign { output: res_sym.clone(), value: true_sym });
            prog.funcs[false_func].blocks[false_block].add_statement(
                Assign { output: res_sym.clone(), value: false_sym });
            if true_func != cur_func || false_func != cur_func {
                // TODO we probably want a better for name for this symbol than whatever res_sym is
                prog.add_local_named(&expr.ty, &res_sym, false_func);
                // the part after the if-else block is split out into a separate continuation
                // function so that we don't have to duplicate this code
                let cont_func = prog.add_func();
                let cont_block = prog.funcs[cont_func].add_block();
                prog.funcs[true_func].blocks[true_block].terminator = JumpFunction(cont_func);
                prog.funcs[false_func].blocks[false_block].terminator = JumpFunction(cont_func);
                Ok((cont_func, cont_block, res_sym))
            } else {
                let cont_block = prog.funcs[cur_func].add_block();
                prog.funcs[true_func].blocks[true_block].terminator = JumpBlock(cont_block);
                prog.funcs[false_func].blocks[false_block].terminator = JumpBlock(cont_block);
                Ok((cur_func, cont_block, res_sym))
            }
        },

        Merge { ref builder, ref value } => {
            let (cur_func, cur_block, builder_sym) = gen_expr(builder, prog, cur_func, cur_block)?;
            let (cur_func, cur_block, elem_sym) = gen_expr(value, prog, cur_func, cur_block)?;
            prog.funcs[cur_func].blocks[cur_block].add_statement(
                DoMerge { builder: builder_sym.clone(), value: elem_sym });
            Ok((cur_func, cur_block, builder_sym))
        },

        Res { ref builder } => {
            let (cur_func, cur_block, builder_sym) = gen_expr(builder, prog, cur_func, cur_block)?;
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(
                GetResult { output: res_sym.clone(), builder: builder_sym });
            Ok((cur_func, cur_block, res_sym))
        },

        NewBuilder => {
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(
                CreateBuilder { output: res_sym.clone(), ty: expr.ty.clone() });
            Ok((cur_func, cur_block, res_sym))
        },

        For { ref iters, ref builder, ref func } => {
            if iters.len() != 1 || iters[0].start.is_some() || iters[0].end.is_some()
            || iters[0].stride.is_some() {
                // TODO support this
                weld_err!("Only single-array loops with null start/end/stride currently supported")?
            }
            let data: &TypedExpr = &iters[0].data;
            if let Lambda { ref params, ref body } = func.kind {
                let (cur_func, cur_block, data_sym) = gen_expr(data, prog, cur_func, cur_block)?;
                let (cur_func, cur_block, builder_sym) = gen_expr(builder, prog, cur_func, cur_block)?;
                let body_func = prog.add_func();
                let body_block = prog.funcs[body_func].add_block();
                prog.add_local_named(&params[0].ty, &params[0].name, body_func);
                prog.add_local_named(&params[1].ty, &params[1].name, body_func);
                prog.funcs[body_func].params.insert(data_sym.clone(), data.ty.clone());
                prog.funcs[body_func].params.insert(builder_sym.clone(), builder.ty.clone());
                let (body_end_func, body_end_block, _) = gen_expr(body, prog, body_func, body_block)?;
                // TODO this is a useless line
                prog.funcs[body_end_func].blocks[body_end_block].terminator = EndFunction;
                let cont_func = prog.add_func();
                let cont_block = prog.funcs[cont_func].add_block();
                prog.funcs[cur_func].blocks[cur_block].terminator = ParallelFor(
                    ParallelForData {
                        data: data_sym,
                        builder: builder_sym.clone(),
                        data_arg: params[1].name.clone(),
                        builder_arg: params[0].name.clone(),
                        body: body_func,
                        cont: cont_func
                    }
                );
                Ok((cont_func, cont_block, builder_sym))
            } else {
                weld_err!("Argument to For was not a Lambda: {}", print_expr(func))
            }
        },

        _ => weld_err!("Unsupported expression: {}", print_expr(expr))
    }
}
