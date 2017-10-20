//! Sequential IR for Weld programs

use std::fmt;
use std::collections::{BTreeMap, HashMap, HashSet};

use super::ast::*;
use super::ast::Type::*;
use super::error::*;
use super::pretty_print::*;
use super::util::SymbolGenerator;

// TODO: make these wrapper types so that you can't pass in the wrong value by mistake
pub type BasicBlockId = usize;
pub type FunctionId = usize;

/// A non-terminating statement inside a basic block.
#[derive(Clone)]
pub enum Statement {
    BinOp {
        output: Symbol,
        op: BinOpKind,
        left: Symbol,
        right: Symbol,
    },
    UnaryOp {
        output: Symbol,
        op: UnaryOpKind,
        child: Symbol,
    },
    Negate { output: Symbol, child: Symbol },
    Broadcast { output: Symbol, child: Symbol },
    Cast {
        output: Symbol,
        child: Symbol,
    },
    Lookup {
        output: Symbol,
        child: Symbol,
        index: Symbol,
    },
    KeyExists {
        output: Symbol,
        child: Symbol,
        key: Symbol,
    },
    Slice {
        output: Symbol,
        child: Symbol,
        index: Symbol,
        size: Symbol,
    },
    Select {
        output: Symbol,
        cond: Symbol,
        on_true: Symbol,
        on_false: Symbol,
    },
    CUDF {
        output: Symbol,
        symbol_name: String,
        args: Vec<Symbol>,
    },
    ToVec { output: Symbol, child: Symbol },
    Length { output: Symbol, child: Symbol },
    Assign { output: Symbol, value: Symbol },
    AssignLiteral { output: Symbol, value: LiteralKind },
    Merge { builder: Symbol, value: Symbol },
    Res { output: Symbol, builder: Symbol },
    NewBuilder {
        output: Symbol,
        arg: Option<Symbol>,
        ty: Type,
    },
    MakeStruct {
        output: Symbol,
        elems: Vec<Symbol>,
    },
    MakeVector {
        output: Symbol,
        elems: Vec<Symbol>,
    },
    GetField {
        output: Symbol,
        value: Symbol,
        index: u32,
    },
}

#[derive(Clone)]
pub struct ParallelForIter {
    pub data: Symbol,
    pub start: Option<Symbol>,
    pub end: Option<Symbol>,
    pub stride: Option<Symbol>,
    pub kind: IterKind,
}

#[derive(Clone)]
pub struct ParallelForData {
    pub data: Vec<ParallelForIter>,
    pub builder: Symbol,
    pub data_arg: Symbol,
    pub builder_arg: Symbol,
    pub idx_arg: Symbol,
    pub body: FunctionId,
    pub cont: FunctionId,
    pub innermost: bool,
    /// If `true`, always invoke parallel runtime for the loop.
    pub always_use_runtime: bool,
}

/// A terminating statement inside a basic block.
#[derive(Clone)]
pub enum Terminator {
    Branch {
        cond: Symbol,
        on_true: BasicBlockId,
        on_false: BasicBlockId,
    },
    JumpBlock(BasicBlockId),
    JumpFunction(FunctionId),
    ProgramReturn(Symbol),
    EndFunction,
    ParallelFor(ParallelForData),
    Crash,
}

/// A basic block inside a SIR program
#[derive(Clone)]
pub struct BasicBlock {
    pub id: BasicBlockId,
    pub statements: Vec<Statement>,
    pub terminator: Terminator,
}

pub struct SirFunction {
    pub id: FunctionId,
    pub params: HashMap<Symbol, Type>,
    pub locals: HashMap<Symbol, Type>,
    pub blocks: Vec<BasicBlock>,
}

impl SirFunction {

    /// Gets the Type for a Symbol in the function. Symbols may be either local variables or
    /// parameters.
    pub fn symbol_type(&self, sym: &Symbol) -> WeldResult<&Type> {
        self.locals.get(sym).map(|s| Ok(s)).unwrap_or_else(|| {
            self.params.get(sym).map(|s| Ok(s)).unwrap_or_else(|| {
                weld_err!("Can't find symbol {}#{}", sym.name, sym.id)
            })
        })
    }
}

pub struct SirProgram {
    /// funcs[0] is the main function
    pub funcs: Vec<SirFunction>,
    pub ret_ty: Type,
    pub top_params: Vec<TypedParameter>,
    sym_gen: SymbolGenerator,
}

impl SirProgram {
    pub fn new(ret_ty: &Type, top_params: &Vec<TypedParameter>) -> SirProgram {
        let mut prog = SirProgram {
            funcs: vec![],
            ret_ty: ret_ty.clone(),
            top_params: top_params.clone(),
            sym_gen: SymbolGenerator::new(),
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
            locals: HashMap::new(),
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
            terminator: Terminator::Crash,
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
            BinOp {
                ref output,
                ref op,
                ref left,
                ref right,
            } => {
                write!(f,
                       "{} = {} {} {}",
                       output,
                       op,
                       left,
                       right)
            }
            UnaryOp {
                ref output,
                ref op,
                ref child,
            } => write!(f, "{} = {}({})", output, op, child),
            Negate {
                ref output,
                ref child,
            } => write!(f, "{} = -{}", output, child),
            Broadcast {
                ref output,
                ref child,
            } => write!(f, "{} = broadcast({})", output, child),
            Cast {
                ref output,
                ref child,
            } => write!(f, "{} = cast({})", output, child),
            Lookup {
                ref output,
                ref child,
                ref index,
            } => write!(f, "{} = lookup({}, {})", output, child, index),
            KeyExists {
                ref output,
                ref child,
                ref key,
            } => write!(f, "{} = keyexists({}, {})", output, child, key),
            Slice {
                ref output,
                ref child,
                ref index,
                ref size,
            } => write!(f, "{} = slice({}, {}, {})", output, child, index, size),
            Select {
                ref output,
                ref cond,
                ref on_true,
                ref on_false,
            } => write!(f, "{} = select({}, {}, {})", output, cond, on_true, on_false),
            ToVec {
                ref output,
                ref child,
            } => write!(f, "{} = toVec({})", output, child),
            Length {
                ref output,
                ref child,
                ..
            } => write!(f, "{} = len({})", output, child),
            Assign {
                ref output,
                ref value,
            } => write!(f, "{} = {}", output, value),
            AssignLiteral {
                ref output,
                ref value,
            } => write!(f, "{} = {}", output, print_literal(value)),
            Merge {
                ref builder,
                ref value,
            } => write!(f, "merge({}, {})", builder, value),
            Res {
                ref output,
                ref builder,
            } => write!(f, "{} = result({})", output, builder),
            NewBuilder {
                ref output,
                ref arg,
                ref ty,
            } => {
                let arg_str = if let Some(ref a) = *arg {
                    a.to_string()
                } else {
                    "".to_string()
                };
                write!(f, "{} = new {}({})", output, print_type(ty), arg_str)
            }
            MakeStruct {
                ref output,
                ref elems,
            } => {
                write!(f,
                       "{} = {}",
                       output,
                       join("{", ",", "}", elems.iter().map(|e| format!("{}", e))))
            }
            MakeVector {
                ref output,
                ref elems,
            } => {
                write!(f,
                       "{} = {}",
                       output,
                       join("[", ", ", "]", elems.iter().map(|e| format!("{}", e))))
            }
            CUDF {
                ref output,
                ref args,
                ref symbol_name,
                ..
            } => {
                write!(f,
                       "{} = cudf[{}]{}",
                       output,
                       symbol_name,
                       join("(", ", ", ")", args.iter().map(|e| format!("{}", e))))
            }
            GetField {
                ref output,
                ref value,
                index,
            } => write!(f, "{} = {}.${}", output, value, index),
        }
    }
}

impl fmt::Display for Terminator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::Terminator::*;
        match *self {
            Branch {
                ref cond,
                ref on_true,
                ref on_false,
            } => write!(f, "branch {} B{} B{}", cond, on_true, on_false),
            ParallelFor(ref pf) => {
                write!(f, "for [")?;
                for iter in &pf.data {
                    write!(f, "{}, ", iter)?;
                }
                write!(f, "] ")?;
                write!(f,
                       "{} {} {} {} F{} F{} {}",
                       pf.builder,
                       pf.builder_arg,
                       pf.idx_arg,
                       pf.data_arg,
                       pf.body,
                       pf.cont,
                       pf.innermost)?;
                Ok(())
            }
            JumpBlock(block) => write!(f, "jump B{}", block),
            JumpFunction(func) => write!(f, "jump F{}", func),
            ProgramReturn(ref sym) => write!(f, "return {}", sym),
            EndFunction => write!(f, "end"),
            Crash => write!(f, "crash"),
        }
    }
}

impl fmt::Display for ParallelForIter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {

        let iterkind = match self.kind {
            IterKind::ScalarIter => "iter",
            IterKind::SimdIter => "simditer",
            IterKind::FringeIter => "fringeiter"
        };

        if self.start.is_some() {
            write!(f,
                   "{}({}, {}, {}, {})",
                   iterkind,
                   self.data,
                   self.start.clone().unwrap(),
                   self.end.clone().unwrap(),
                   self.stride.clone().unwrap())
        } else if self.kind != IterKind::ScalarIter {
            write!(f, "{}({})", iterkind, self.data)
        } else {
            write!(f, "{}", self.data)
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

/// Recursive helper function for sir_param_correction. `env` contains the symbol to type mappings
/// that have been defined previously in the program. Any symbols that need to be passed in
/// as closure parameters to func_id will be added to `closure` (so that `func_id`'s
/// callers can also add these symbols to their parameters list, if necessary).
/// `visited` contains functions we have already seen on the way down the function call tree,
/// to prevent infinite recursion when there are loops.
fn sir_param_correction_helper(prog: &mut SirProgram,
                               func_id: FunctionId,
                               env: &mut HashMap<Symbol, Type>,
                               closure: &mut HashSet<Symbol>,
                               visited: &mut HashSet<FunctionId>) {
    // this is needed for cases where params are added outside of sir_param_correction and are not
    // based on variable reads in the function (e.g. in the Iterate case);
    // and when there are loops in the call graph (also in the Iterate case)
    for (name, _) in &prog.funcs[func_id].params {
        closure.insert(name.clone());
    }
    if !visited.insert(func_id) {
        return;
    }
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
                // push any existing symbols that are used (but not assigned) by the statement
                BinOp {
                    ref left,
                    ref right,
                    ..
                } => {
                    vars.push(left.clone());
                    vars.push(right.clone());
                }
                UnaryOp {
                    ref child,
                    ..
                } => {
                    vars.push(child.clone());
                }
                Cast { ref child, .. } => {
                    vars.push(child.clone());
                }
                Negate { ref child, .. } => {
                    vars.push(child.clone());
                }
                Broadcast { ref child, .. } => {
                    vars.push(child.clone());
                }
                Lookup {
                    ref child,
                    ref index,
                    ..
                } => {
                    vars.push(child.clone());
                    vars.push(index.clone());
                }
                KeyExists { ref child, ref key, .. } => {
                    vars.push(child.clone());
                    vars.push(key.clone());
                }
                Slice {
                    ref child,
                    ref index,
                    ref size,
                    ..
                } => {
                    vars.push(child.clone());
                    vars.push(index.clone());
                    vars.push(size.clone());
                }
                Select {
                    ref cond,
                    ref on_true,
                    ref on_false,
                    ..
                } => {
                    vars.push(cond.clone());
                    vars.push(on_true.clone());
                    vars.push(on_false.clone());
                }
                ToVec { ref child, .. } => {
                    vars.push(child.clone());
                }
                Length { ref child, .. } => {
                    vars.push(child.clone());
                }
                Assign { ref value, .. } => vars.push(value.clone()),
                Merge {
                    ref builder,
                    ref value,
                } => {
                    vars.push(builder.clone());
                    vars.push(value.clone());
                }
                Res { ref builder, .. } => vars.push(builder.clone()),
                GetField { ref value, .. } => vars.push(value.clone()),
                AssignLiteral { .. } => {}
                NewBuilder { ref arg, .. } => {
                    if let Some(ref a) = *arg {
                        vars.push(a.clone());
                    }
                }
                MakeStruct { ref elems, .. } => {
                    for elem in elems {
                        vars.push(elem.clone());
                    }
                }
                MakeVector { ref elems, .. } => {
                    for elem in elems {
                        vars.push(elem.clone());
                    }
                }
                CUDF { ref args, .. } => {
                    for arg in args {
                        vars.push(arg.clone());
                    }
                }
            }
        }
        use self::Terminator::*;
        match block.terminator {
            // push any existing symbols that are used by the terminator
            Branch { ref cond, .. } => {
                vars.push(cond.clone());
            }
            ProgramReturn(ref sym) => {
                vars.push(sym.clone());
            }
            ParallelFor(ref pf) => {
                for iter in pf.data.iter() {
                    vars.push(iter.data.clone());
                    if iter.start.is_some() {
                        vars.push(iter.start.clone().unwrap());
                        vars.push(iter.end.clone().unwrap());
                        vars.push(iter.stride.clone().unwrap());
                    }
                }
                vars.push(pf.builder.clone());
            }
            JumpBlock(_) => {}
            JumpFunction(_) => {}
            EndFunction => {}
            Crash => {}
        }
        for var in &vars {
            if prog.funcs[func_id].locals.get(&var) == None {
                prog.funcs[func_id]
                    .params
                    .insert(var.clone(), env.get(&var).unwrap().clone());
                closure.insert(var.clone());
            }
        }
        let mut inner_closure = HashSet::new();
        match block.terminator {
            // make a recursive call for other functions referenced by the terminator
            ParallelFor(ref pf) => {
                sir_param_correction_helper(prog, pf.body, env, &mut inner_closure, visited);
                sir_param_correction_helper(prog, pf.cont, env, &mut inner_closure, visited);
            }
            JumpFunction(jump_func) => {
                sir_param_correction_helper(prog, jump_func, env, &mut inner_closure, visited);
            }
            Branch { .. } => {}
            JumpBlock(_) => {}
            ProgramReturn(_) => {}
            EndFunction => {}
            Crash => {}
        }
        for var in inner_closure {
            if prog.funcs[func_id].locals.get(&var) == None {
                prog.funcs[func_id]
                    .params
                    .insert(var.clone(), env.get(&var).unwrap().clone());
                closure.insert(var.clone());
            }
        }
    }
}

/// gen_expr may result in the use of symbols across function boundaries,
/// so ast_to_sir calls sir_param_correction to correct function parameters
/// to ensure that such symbols (the closure) are passed in as parameters.
/// Can be safely called multiple times -- only the necessary param corrections
/// will be performed.
fn sir_param_correction(prog: &mut SirProgram) -> WeldResult<()> {
    let mut env = HashMap::new();
    let mut closure = HashSet::new();
    let mut visited = HashSet::new();
    sir_param_correction_helper(prog, 0, &mut env, &mut closure, &mut visited);
    let ref func = prog.funcs[0];
    for name in closure {
        if func.params.get(&name) == None {
            weld_err!("Unbound symbol {}#{}", name.name, name.id)?;
        }
    }
    Ok(())
}

/// Convert an AST to a SIR program. Symbols must be unique in expr.
pub fn ast_to_sir(expr: &TypedExpr, multithreaded: bool) -> WeldResult<SirProgram> {
    if let ExprKind::Lambda { ref params, ref body } = expr.kind {
        let mut prog = SirProgram::new(&expr.ty, params);
        prog.sym_gen = SymbolGenerator::from_expression(expr);
        for tp in params {
            prog.funcs[0].params.insert(tp.name.clone(), tp.ty.clone());
        }
        let first_block = prog.funcs[0].add_block();
        let (res_func, res_block, res_sym) = gen_expr(body, &mut prog, 0, first_block, multithreaded)?;
        prog.funcs[res_func].blocks[res_block].terminator = Terminator::ProgramReturn(res_sym);
        sir_param_correction(&mut prog)?;
        // second call is necessary in the case where there are loops in the call graph, since
        // some parameter dependencies may not have been propagated through back edges
        sir_param_correction(&mut prog)?;
        Ok((prog))
    } else {
        weld_err!("Expression passed to ast_to_sir was not a Lambda")
    }
}

/// Generate code to compute the expression `expr` starting at the current tail of `cur_block`,
/// possibly creating new basic blocks and functions in the process. Return the function and
/// basic block that the expression will be ready in, and its symbol therein.
fn gen_expr(expr: &TypedExpr,
            prog: &mut SirProgram,
            cur_func: FunctionId,
            cur_block: BasicBlockId,
            multithreaded: bool)
            -> WeldResult<(FunctionId, BasicBlockId, Symbol)> {
    use self::Statement::*;
    use self::Terminator::*;
    match expr.kind {
        ExprKind::Ident(ref sym) => Ok((cur_func, cur_block, sym.clone())),

        ExprKind::Literal(lit) => {
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(AssignLiteral {
                                                                     output: res_sym.clone(),
                                                                     value: lit,
                                                                 });
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Let {
            ref name,
            ref value,
            ref body,
        } => {
            let (cur_func, cur_block, val_sym) = gen_expr(value, prog, cur_func, cur_block, multithreaded)?;
            prog.add_local_named(&value.ty, name, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(Assign {
                                                                     output: name.clone(),
                                                                     value: val_sym,
                                                                 });
            let (cur_func, cur_block, res_sym) = gen_expr(body, prog, cur_func, cur_block, multithreaded)?;
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::BinOp {
            kind,
            ref left,
            ref right,
        } => {
            let (cur_func, cur_block, left_sym) = gen_expr(left, prog, cur_func, cur_block, multithreaded)?;
            let (cur_func, cur_block, right_sym) = gen_expr(right, prog, cur_func, cur_block, multithreaded)?;
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(BinOp {
                                                                     output: res_sym.clone(),
                                                                     op: kind,
                                                                     left: left_sym,
                                                                     right: right_sym,
                                                                 });
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::UnaryOp {
            kind,
            ref value,
        } => {
            let (cur_func, cur_block, value_sym) = gen_expr(value, prog, cur_func, cur_block, multithreaded)?;
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(UnaryOp {
                output: res_sym.clone(),
                op: kind,
                child: value_sym,
            });
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Negate(ref child_expr) => {
            let (cur_func, cur_block, child_sym) = gen_expr(child_expr, prog, cur_func, cur_block, multithreaded)?;
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(Negate {
                                                                     output: res_sym.clone(),
                                                                     child: child_sym,
                                                                 });
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Broadcast(ref child_expr) => {
            let (cur_func, cur_block, child_sym) = gen_expr(child_expr, prog, cur_func, cur_block, multithreaded)?;
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(Broadcast {
                                                                     output: res_sym.clone(),
                                                                     child: child_sym,
                                                                 });
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Cast { ref child_expr, .. } => {
            let (cur_func, cur_block, child_sym) = gen_expr(child_expr, prog, cur_func, cur_block, multithreaded)?;
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(Cast {
                                                                     output: res_sym.clone(),
                                                                     child: child_sym,
                                                                 });
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Lookup {
            ref data,
            ref index,
        } => {
            let (cur_func, cur_block, data_sym) = gen_expr(data, prog, cur_func, cur_block, multithreaded)?;
            let (cur_func, cur_block, index_sym) = gen_expr(index, prog, cur_func, cur_block, multithreaded)?;
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(Lookup {
                                                                     output: res_sym.clone(),
                                                                     child: data_sym,
                                                                     index: index_sym.clone(),
                                                                 });
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::KeyExists { ref data, ref key } => {
            let (cur_func, cur_block, data_sym) = gen_expr(data, prog, cur_func, cur_block, multithreaded)?;
            let (cur_func, cur_block, key_sym) = gen_expr(key, prog, cur_func, cur_block, multithreaded)?;
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(KeyExists {
                                                                     output: res_sym.clone(),
                                                                     child: data_sym,
                                                                     key: key_sym.clone(),
                                                                 });
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Slice {
            ref data,
            ref index,
            ref size,
        } => {
            let (cur_func, cur_block, data_sym) = gen_expr(data, prog, cur_func, cur_block, multithreaded)?;
            let (cur_func, cur_block, index_sym) = gen_expr(index, prog, cur_func, cur_block, multithreaded)?;
            let (cur_func, cur_block, size_sym) = gen_expr(size, prog, cur_func, cur_block, multithreaded)?;
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(Slice {
                                                                     output: res_sym.clone(),
                                                                     child: data_sym,
                                                                     index: index_sym.clone(),
                                                                     size: size_sym.clone(),
                                                                 });
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Select {
            ref cond,
            ref on_true,
            ref on_false,
        } => {
            let (cur_func, cur_block, cond_sym) = gen_expr(cond, prog, cur_func, cur_block, multithreaded)?;
            let (cur_func, cur_block, true_sym) = gen_expr(on_true, prog, cur_func, cur_block, multithreaded)?;
            let (cur_func, cur_block, false_sym) = gen_expr(on_false, prog, cur_func, cur_block, multithreaded)?;
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(Select {
                                                                     output: res_sym.clone(),
                                                                     cond: cond_sym,
                                                                     on_true: true_sym.clone(),
                                                                     on_false: false_sym.clone(),
                                                                 });
            Ok((cur_func, cur_block, res_sym))
        }
        ExprKind::ToVec { ref child_expr } => {
            let (cur_func, cur_block, child_sym) = gen_expr(child_expr, prog, cur_func, cur_block, multithreaded)?;
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(ToVec {
                                                                     output: res_sym.clone(),
                                                                     child: child_sym,
                                                                 });
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Length { ref data } => {
            let (cur_func, cur_block, data_sym) = gen_expr(data, prog, cur_func, cur_block, multithreaded)?;
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(Length {
                                                                     output: res_sym.clone(),
                                                                     child: data_sym,
                                                                 });
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::If {
            ref cond,
            ref on_true,
            ref on_false,
        } => {
            let (cur_func, cur_block, cond_sym) = gen_expr(cond, prog, cur_func, cur_block, multithreaded)?;
            let true_block = prog.funcs[cur_func].add_block();
            let false_block = prog.funcs[cur_func].add_block();
            prog.funcs[cur_func].blocks[cur_block].terminator = Branch {
                cond: cond_sym,
                on_true: true_block,
                on_false: false_block,
            };
            let (true_func, true_block, true_sym) = gen_expr(on_true, prog, cur_func, true_block, multithreaded)?;
            let (false_func, false_block, false_sym) =
                gen_expr(on_false, prog, cur_func, false_block, multithreaded)?;
            let res_sym = prog.add_local(&expr.ty, true_func);
            prog.funcs[true_func].blocks[true_block].add_statement(Assign {
                                                                       output: res_sym.clone(),
                                                                       value: true_sym,
                                                                   });
            prog.funcs[false_func].blocks[false_block].add_statement(Assign {
                                                                         output: res_sym.clone(),
                                                                         value: false_sym,
                                                                     });
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
        }

        ExprKind::Iterate {
            ref initial,
            ref update_func,
        } => {
            // Generate the intial value.
            let (cur_func, cur_block, initial_sym) = gen_expr(initial, prog, cur_func, cur_block, multithreaded)?;

            // Pull out the argument name and function body and validate that things type-check.
            let argument_sym;
            let func_body;
            match update_func.kind {
                ExprKind::Lambda { ref params, ref body } if params.len() == 1 => {
                    argument_sym = &params[0].name;
                    func_body = body;
                    if params[0].ty != initial.ty {
                        return weld_err!("Wrong argument type for body of Iterate");
                    }
                    if func_body.ty != Struct(vec![initial.ty.clone(), Scalar(ScalarKind::Bool)]) {
                        return weld_err!("Wrong return type for body of Iterate");
                    }
                    prog.add_local_named(&params[0].ty, argument_sym, cur_func);
                }
                _ => return weld_err!("Argument of Iterate was not a Lambda")
            }
            prog.funcs[cur_func].blocks[cur_block].add_statement(
                Assign { output: argument_sym.clone(), value: initial_sym });

            // Check whether the function's body contains any parallel loops. If so, we should put the loop body
            // in a new function because we'll need to jump back to it from continuations. If not, we can just
            // make the loop body be another basic block in the current function.
            let parallel_body = contains_parallel_expressions(func_body);
            let body_start_func = if parallel_body {
                let new_func = prog.add_func();
                new_func
            } else {
                cur_func
            };

            let body_start_block = prog.funcs[body_start_func].add_block();

            // Jump to where the body starts
            if parallel_body {
                prog.funcs[cur_func].blocks[cur_block].terminator = JumpFunction(body_start_func);
            } else {
                prog.funcs[cur_func].blocks[cur_block].terminator = JumpBlock(body_start_block);
            }

            // Generate the loop's body, which will work on argument_sym and produce result_sym.
            // The type of result_sym will be {ArgType, bool} and we will repeat the body if the bool is true.
            let (body_end_func, body_end_block, result_sym) =
                gen_expr(func_body, prog, body_start_func, body_start_block, multithreaded)?;

            // After the body, unpack the {state, bool} struct into symbols argument_sym and continue_sym.
            let continue_sym = prog.add_local(&Scalar(ScalarKind::Bool), body_end_func);
            if parallel_body {
                // this is needed because sir_param_correction does not add variables only used
                // on the LHS of assignments to the params list
                prog.funcs[body_end_func].params.insert(argument_sym.clone(), initial.ty.clone());
            }
            prog.funcs[body_end_func].blocks[body_end_block].add_statement(
                GetField { output: argument_sym.clone(), value: result_sym.clone(), index: 0 });
            prog.funcs[body_end_func].blocks[body_end_block].add_statement(
                GetField { output: continue_sym.clone(), value: result_sym.clone(), index: 1 });

            // Create two more blocks so we can branch on continue_sym
            let repeat_block = prog.funcs[body_end_func].add_block();
            let finish_block = prog.funcs[body_end_func].add_block();
            prog.funcs[body_end_func].blocks[body_end_block].terminator =
                Branch { cond: continue_sym, on_true: repeat_block, on_false: finish_block };

            // If we had a parallel body, repeat_block must do a JumpFunction to get back to body_start_func;
            // otherwise it can just do a normal JumpBlock since it should be in the same function.
            if parallel_body {
                assert!(body_end_func != body_start_func);
                prog.funcs[body_end_func].blocks[repeat_block].terminator = JumpFunction(body_start_func);
            } else {
                assert!(body_end_func == cur_func && body_start_func == cur_func);
                prog.funcs[body_end_func].blocks[repeat_block].terminator = JumpBlock(body_start_block);
            }

            // In either case, our final value is available in finish_block.
            Ok((body_end_func, finish_block, argument_sym.clone()))
        }

        ExprKind::Merge {
            ref builder,
            ref value,
        } => {
            let (cur_func, cur_block, builder_sym) = gen_expr(builder, prog, cur_func, cur_block, multithreaded)?;
            let (cur_func, cur_block, elem_sym) = gen_expr(value, prog, cur_func, cur_block, multithreaded)?;
            prog.funcs[cur_func].blocks[cur_block].add_statement(Merge {
                                                                     builder: builder_sym.clone(),
                                                                     value: elem_sym,
                                                                 });
            Ok((cur_func, cur_block, builder_sym))
        }

        ExprKind::Res { ref builder } => {
            let (cur_func, cur_block, builder_sym) = gen_expr(builder, prog, cur_func, cur_block, multithreaded)?;
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(Res {
                                                                     output: res_sym.clone(),
                                                                     builder: builder_sym,
                                                                 });
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::NewBuilder(ref arg) => {
            let (cur_func, cur_block, arg_sym) = if let Some(ref a) = *arg {
                let (cur_func, cur_block, arg_sym) = gen_expr(a, prog, cur_func, cur_block, multithreaded)?;
                (cur_func, cur_block, Some(arg_sym))
            } else {
                (cur_func, cur_block, None)
            };
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(NewBuilder {
                                                                     output: res_sym.clone(),
                                                                     arg: arg_sym,
                                                                     ty: expr.ty.clone(),
                                                                 });
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::MakeStruct { ref elems } => {
            let mut syms = vec![];
            let (mut cur_func, mut cur_block, mut sym) =
                gen_expr(&elems[0], prog, cur_func, cur_block, multithreaded)?;
            syms.push(sym);
            for elem in elems.iter().skip(1) {
                let r = gen_expr(elem, prog, cur_func, cur_block, multithreaded)?;
                cur_func = r.0;
                cur_block = r.1;
                sym = r.2;
                syms.push(sym);
            }
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(MakeStruct {
                                                                     output: res_sym.clone(),
                                                                     elems: syms,
                                                                 });
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::MakeVector { ref elems } => {
            let mut syms = vec![];
            let mut cur_func = cur_func;
            let mut cur_block = cur_block;
            for elem in elems.iter() {
                let r = gen_expr(elem, prog, cur_func, cur_block, multithreaded)?;
                cur_func = r.0;
                cur_block = r.1;
                let sym = r.2;
                syms.push(sym);
            }
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(MakeVector {
                                                                     output: res_sym.clone(),
                                                                     elems: syms
                                                                 });
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::CUDF {
            ref sym_name,
            ref args,
            ..
        } => {
            let mut syms = vec![];
            let mut cur_func = cur_func;
            let mut cur_block = cur_block;
            for arg in args.iter() {
                let r = gen_expr(arg, prog, cur_func, cur_block, multithreaded)?;
                cur_func = r.0;
                cur_block = r.1;
                let sym = r.2;
                syms.push(sym);
            }
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(CUDF {
                                                                     output: res_sym.clone(),
                                                                     args: syms,
                                                                     symbol_name: sym_name.clone(),
                                                                 });
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::GetField { ref expr, index } => {
            let (cur_func, cur_block, struct_sym) = gen_expr(expr, prog, cur_func, cur_block, multithreaded)?;
            let field_ty = match expr.ty {
                super::ast::Type::Struct(ref v) => &v[index as usize],
                _ => {
                    weld_err!("Internal error: tried to get field of type {}",
                              print_type(&expr.ty))?
                }
            };
            let res_sym = prog.add_local(&field_ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(GetField {
                                                                     output: res_sym.clone(),
                                                                     value: struct_sym,
                                                                     index: index,
                                                                 });
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::For {
            ref iters,
            ref builder,
            ref func,
        } => {
            if let ExprKind::Lambda {
                       ref params,
                       ref body,
                   } = func.kind {
                let (cur_func, cur_block, builder_sym) =
                    gen_expr(builder, prog, cur_func, cur_block, multithreaded)?;
                let body_func = prog.add_func();
                let body_block = prog.funcs[body_func].add_block();
                prog.add_local_named(&params[0].ty, &params[0].name, body_func);
                prog.add_local_named(&params[1].ty, &params[1].name, body_func);
                prog.add_local_named(&params[2].ty, &params[2].name, body_func);
                prog.funcs[body_func]
                    .params
                    .insert(builder_sym.clone(), builder.ty.clone());
                let mut cur_func = cur_func;
                let mut cur_block = cur_block;
                let mut pf_iters: Vec<ParallelForIter> = Vec::new();
                for iter in iters.iter() {
                    let data_res = gen_expr(&iter.data, prog, cur_func, cur_block, multithreaded)?;
                    cur_func = data_res.0;
                    cur_block = data_res.1;
                    prog.funcs[body_func]
                        .params
                        .insert(data_res.2.clone(), iter.data.ty.clone());
                    let start_sym = if iter.start.is_some() {
                        // TODO is there a cleaner way to do this?
                        let start_expr = match iter.start {
                            Some(ref e) => e,
                            _ => weld_err!("Can't reach this")?,
                        };
                        let start_res = gen_expr(&start_expr, prog, cur_func, cur_block, multithreaded)?;
                        cur_func = start_res.0;
                        cur_block = start_res.1;
                        prog.funcs[body_func]
                            .params
                            .insert(start_res.2.clone(), start_expr.ty.clone());
                        Some(start_res.2)
                    } else {
                        None
                    };
                    let end_sym = if iter.end.is_some() {
                        let end_expr = match iter.end {
                            Some(ref e) => e,
                            _ => weld_err!("Can't reach this")?,
                        };
                        let end_res = gen_expr(&end_expr, prog, cur_func, cur_block, multithreaded)?;
                        cur_func = end_res.0;
                        cur_block = end_res.1;
                        prog.funcs[body_func]
                            .params
                            .insert(end_res.2.clone(), end_expr.ty.clone());
                        Some(end_res.2)
                    } else {
                        None
                    };
                    let stride_sym = if iter.stride.is_some() {
                        let stride_expr = match iter.stride {
                            Some(ref e) => e,
                            _ => weld_err!("Can't reach this")?,
                        };
                        let stride_res = gen_expr(&stride_expr, prog, cur_func, cur_block, multithreaded)?;
                        cur_func = stride_res.0;
                        cur_block = stride_res.1;
                        prog.funcs[body_func]
                            .params
                            .insert(stride_res.2.clone(), stride_expr.ty.clone());
                        Some(stride_res.2)
                    } else {
                        None
                    };
                    pf_iters.push(ParallelForIter {
                                      data: data_res.2,
                                      start: start_sym,
                                      end: end_sym,
                                      stride: stride_sym,
                                      kind: iter.kind.clone(),
                                  });
                }
                let (body_end_func, body_end_block, _) =
                    gen_expr(body, prog, body_func, body_block, multithreaded)?;
                prog.funcs[body_end_func].blocks[body_end_block].terminator = EndFunction;
                let cont_func = prog.add_func();
                let cont_block = prog.funcs[cont_func].add_block();
                let mut is_innermost = true;
                body.traverse(&mut |ref e| if let ExprKind::For { .. } = e.kind {
                                       is_innermost = false;
                                   });
                prog.funcs[cur_func].blocks[cur_block].terminator =
                    ParallelFor(ParallelForData {
                                    data: pf_iters,
                                    builder: builder_sym.clone(),
                                    data_arg: params[2].name.clone(),
                                    builder_arg: params[0].name.clone(),
                                    idx_arg: params[1].name.clone(),
                                    body: body_func,
                                    cont: cont_func,
                                    innermost: is_innermost,
                                    always_use_runtime: expr.annotations.always_use_runtime(),
                                });
                Ok((cont_func, cont_block, builder_sym))
            } else {
                weld_err!("Argument to For was not a Lambda: {}", print_expr(func))
            }
        }

        _ => weld_err!("Unsupported expression: {}", print_expr(expr)),
    }
}

/// Return true if an expression contains parallel for operators
fn contains_parallel_expressions(expr: &TypedExpr) -> bool {
    let mut found = false;
    expr.traverse(&mut |ref e| {
        if let ExprKind::For { .. } = e.kind {
            found = true;
        }
    });
    found
}

fn join<T: Iterator<Item = String>>(start: &str, sep: &str, end: &str, strings: T) -> String {
    let mut res = String::new();
    res.push_str(start);
    for (i, s) in strings.enumerate() {
        if i > 0 {
            res.push_str(sep);
        }
        res.push_str(&s);
    }
    res.push_str(end);
    res
}
