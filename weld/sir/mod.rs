//! Sequential IR for Weld programs

use std::fmt;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::collections::hash_map::Entry;

use std::vec;

use super::ast::*;
use super::ast::Type::*;
use super::error::*;
use super::util::{SymbolGenerator, join};

extern crate fnv;

pub mod optimizations;

// TODO: make these wrapper types so that you can't pass in the wrong value by mistake
pub type BasicBlockId = usize;
pub type FunctionId = usize;

/// A non-terminating statement inside a basic block.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum StatementKind {
    Assign(Symbol),
    AssignLiteral(LiteralKind),
    BinOp {
        op: BinOpKind,
        left: Symbol,
        right: Symbol,
    },
    Broadcast(Symbol),
    Cast(Symbol, Type),
    CUDF {
        symbol_name: String,
        args: Vec<Symbol>,
    },
    GetField {
        value: Symbol,
        index: u32,
    },
    KeyExists {
        child: Symbol,
        key: Symbol,
    },
    Length(Symbol),
    Lookup {
        child: Symbol,
        index: Symbol,
    },
    OptLookup {
        child: Symbol,
        index: Symbol,
    },
    MakeStruct(Vec<Symbol>),
    MakeVector(Vec<Symbol>),
    Merge { builder: Symbol, value: Symbol },
    Negate(Symbol),
    Not(Symbol),
    NewBuilder {
        arg: Option<Symbol>,
        ty: Type,
    },
    ParallelFor(ParallelForData),
    Res(Symbol),
    Select {
        cond: Symbol,
        on_true: Symbol,
        on_false: Symbol,
    },
    Slice {
        child: Symbol,
        index: Symbol,
        size: Symbol,
    },
    Sort {
        child: Symbol,
        keyfunc: SirFunction,
    },
    Serialize(Symbol),
    Deserialize(Symbol),
    ToVec(Symbol),
    UnaryOp {
        op: UnaryOpKind,
        child: Symbol,
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ParallelForIter {
    pub data: Symbol,
    pub start: Option<Symbol>,
    pub end: Option<Symbol>,
    pub stride: Option<Symbol>,
    pub kind: IterKind,
    // NdIter specific fields
    pub strides: Option<Symbol>,
    pub shape: Option<Symbol>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ParallelForData {
    pub data: Vec<ParallelForIter>,
    pub builder: Symbol,
    pub data_arg: Symbol,
    pub builder_arg: Symbol,
    pub idx_arg: Symbol,
    pub body: FunctionId,
    pub innermost: bool,
}

impl StatementKind {
    pub fn children(&self) -> vec::IntoIter<&Symbol> {
        use self::StatementKind::*;
        let mut vars = vec![];
        match *self {
            // push any existing symbols that are used (but not assigned) by the statement
            BinOp {
                ref left,
                ref right,
                ..
            } => {
                vars.push(left);
                vars.push(right);
            }
            ParallelFor(ref data) => {
                vars.push(&data.builder);
                // vars.push(&data.data_arg);
                // vars.push(&data.builder_arg);
                // vars.push(&data.idx_arg);
                for iter in data.data.iter() {
                    vars.push(&iter.data);
                    if iter.shape.is_some() {
                        vars.push(iter.start.as_ref().unwrap());
                        vars.push(iter.end.as_ref().unwrap());
                        vars.push(iter.stride.as_ref().unwrap());
                        vars.push(iter.shape.as_ref().unwrap());
                        vars.push(iter.strides.as_ref().unwrap());
                    } else if iter.start.is_some() {
                        vars.push(iter.start.as_ref().unwrap());
                        vars.push(iter.end.as_ref().unwrap());
                        vars.push(iter.stride.as_ref().unwrap());
                    }
                }
            }
            UnaryOp {
                ref child,
                ..
            } => {
                vars.push(child);
            }
            Cast(ref child, _) => {
                vars.push(child);
            }
            Negate(ref child) => {
                vars.push(child);
            }
            Not(ref child) => {
                vars.push(child);
            }
            Broadcast(ref child) => {
                vars.push(child);
            }
            Serialize(ref child) => {
                vars.push(child);
            }
            Deserialize(ref child) => {
                vars.push(child);
            }
            Lookup {
                ref child,
                ref index,
            } => {
                vars.push(child);
                vars.push(index);
            }
            OptLookup {
                ref child,
                ref index,
            } => {
                vars.push(child);
                vars.push(index);
            }
            KeyExists { ref child, ref key } => {
                vars.push(child);
                vars.push(key);
            }
            Slice {
                ref child,
                ref index,
                ref size,
            } => {
                vars.push(child);
                vars.push(index);
                vars.push(size);
            }
            Sort {
                ref child,
                ..
            } => {
                vars.push(child);
            }
            Select {
                ref cond,
                ref on_true,
                ref on_false,
            } => {
                vars.push(cond);
                vars.push(on_true);
                vars.push(on_false);
            }
            ToVec(ref child) => {
                vars.push(child);
            }
            Length(ref child) => {
                vars.push(child);
            }
            Assign(ref value) => {
                vars.push(value);
            }
            Merge {
                ref builder,
                ref value,
            } => {
                vars.push(builder);
                vars.push(value);
            }
            Res(ref builder) => vars.push(builder),
            GetField { ref value, .. } => vars.push(value),
            AssignLiteral { .. } => {}
            NewBuilder { ref arg, .. } => {
                if let Some(ref a) = *arg {
                    vars.push(a);
                }
            }
            MakeStruct(ref elems) => {
                for elem in elems {
                    vars.push(elem);
                }
            }
            MakeVector(ref elems) => {
                for elem in elems {
                    vars.push(elem);
                }
            }
            CUDF {
                ref args,
                ..
            } => {
                for arg in args {
                    vars.push(arg);
                }
            }
        }
        vars.into_iter()
    }
}

/// A single statement in the SIR, with a RHS statement kind and an optional LHS output symbol.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Statement {
    pub output: Option<Symbol>,
    pub kind: StatementKind,
}

impl Statement {
    pub fn new(output: Option<Symbol>, kind: StatementKind) -> Statement {
        Statement {
            output: output,
            kind: kind,
        }
    }
}

/// Wrapper type to add statements into a program. This object prevents statements from being
/// produced more than once.

/// A site in the program, identified via a `FunctionId` and `BasicBlockId`
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ProgramSite(FunctionId, BasicBlockId);

/// Maps generated statements to the symbol representing the output of that statement in a given
/// site.
type SiteSymbolMap = fnv::FnvHashMap<StatementKind, Symbol>;

struct StatementTracker {
    generated: fnv::FnvHashMap<ProgramSite,SiteSymbolMap>,
}

impl StatementTracker {

    pub fn new() -> StatementTracker {
        StatementTracker {
            generated: fnv::FnvHashMap::default(),
        }
    }

    /// Returns a symbol holding the value of the given `StatementKind` in `(func, block)`. If a
    /// symbol representing this statement does not exist, the statement is added to the program
    /// and a new `Symbol` is returned.
    ///
    /// This function should not be used for statements with _named_ parameters (e.g., identifiers,
    /// parameters in a `Lambda`, or names bound using a `Let` statement.)!
    fn symbol_for_statement(&mut self,
                            prog: &mut SirProgram,
                            func: FunctionId,
                            block: BasicBlockId,
                            sym_ty: &Type,
                            kind: StatementKind) -> Symbol {

        use sir::StatementKind::CUDF;

        let site = ProgramSite(func, block);
        let map = self.generated.entry(site).or_insert(fnv::FnvHashMap::default());

        // CUDFs are the only functions that can have side-effects so we always need to give them
        // a new name.
        if let CUDF { .. } = kind {
            let res_sym = prog.add_local(sym_ty, func);
            prog.funcs[func].blocks[block].add_statement(Statement::new(Some(res_sym.clone()), kind));
            return res_sym;
        }

        // Return the symbol to use.
        match map.entry(kind.clone()) {
            Entry::Occupied(ent) => {
                ent.get().clone()
            }
            Entry::Vacant(ent) => {
                let res_sym = prog.add_local(sym_ty, func);
                prog.funcs[func].blocks[block].add_statement(Statement::new(Some(res_sym.clone()), kind));
                ent.insert(res_sym.clone());
                res_sym
            }
        }
    }

    /// Adds a Statement with a named statement.
    fn named_symbol_for_statement(&mut self,
                                  prog: &mut SirProgram,
                                  func: FunctionId,
                                  block: BasicBlockId,
                                  sym_ty: &Type,
                                  kind: StatementKind,
                                  named_sym: Symbol) {

        let site = ProgramSite(func, block);
        let map = self.generated.entry(site).or_insert(fnv::FnvHashMap::default());

        prog.add_local_named(sym_ty, &named_sym, func);
        prog.funcs[func].blocks[block].add_statement(Statement::new(Some(named_sym.clone()), kind.clone()));
        map.insert(kind, named_sym.clone());
    }
}


/// A terminating statement inside a basic block.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Terminator {
    Branch {
        cond: Symbol,
        on_true: BasicBlockId,
        on_false: BasicBlockId,
    },
    JumpBlock(BasicBlockId),
    ProgramReturn(Symbol),
    EndFunction(Symbol),
    Crash,
}

impl Terminator {
    /// Returns Symbols that the `Terminator` depends on.
    pub fn children(&self) -> vec::IntoIter<&Symbol> {
        use self::Terminator::*;
        let mut vars = vec![];
        match *self {
            Branch { ref cond, .. } => {
                vars.push(cond);
            }
            ProgramReturn(ref sym) => {
                vars.push(sym);
            }
            EndFunction(ref sym) => {
                vars.push(&sym)
            }
            Crash => (),
            JumpBlock(_) => (),
        };
        vars.into_iter()
    }
}

/// A basic block inside a SIR program
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct BasicBlock {
    pub id: BasicBlockId,
    pub statements: Vec<Statement>,
    pub terminator: Terminator,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SirFunction {
    pub id: FunctionId,
    pub params: BTreeMap<Symbol, Type>,
    pub locals: BTreeMap<Symbol, Type>,
    // Local variables that are used for looping.
    //
    // This will be an empty vector if `loop_body` is false. If loop_body is true, the variables
    // that appear here are guaranteed to also be in `locals`.
    pub loop_variables: Vec<Symbol>,
    pub blocks: Vec<BasicBlock>,
    pub return_type: Type,
    pub loop_body: bool,
}

impl SirFunction {
    /// Gets the Type for a Symbol in the function. Symbols may be either local variables or
    /// parameters.
    pub fn symbol_type(&self, sym: &Symbol) -> WeldResult<&Type> {
        self.locals.get(sym).map(|s| Ok(s)).unwrap_or_else(|| {
            self.params.get(sym).map(|s| Ok(s)).unwrap_or_else(|| {
                compile_err!("Can't find symbol {}", sym.to_string())
            })
        })
    }
}

pub struct SirProgram {
    /// funcs[0] is the main function
    pub funcs: Vec<SirFunction>,
    pub ret_ty: Type,
    pub top_params: Vec<Parameter>,
    sym_gen: SymbolGenerator,
}

impl SirProgram {
    pub fn new(ret_ty: &Type, top_params: &Vec<Parameter>) -> SirProgram {
        let mut prog = SirProgram {
            funcs: vec![],
            ret_ty: ret_ty.clone(),
            top_params: top_params.clone(),
            sym_gen: SymbolGenerator::new(),
        };
        // Add the main function.
        prog.add_func();
        prog
    }

    pub fn add_func(&mut self) -> FunctionId {
        let func = SirFunction {
            id: self.funcs.len(),
            params: BTreeMap::new(),
            blocks: vec![],
            locals: BTreeMap::new(),
            loop_variables: vec![],
            return_type: Unknown,
            loop_body: false,
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

    pub fn add_loop_variable(&mut self, sym: &Symbol, func: FunctionId) {
        self.funcs[func].loop_variables.push(sym.clone());
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

impl fmt::Display for StatementKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::StatementKind::*;
        match *self {
            Assign(ref value) => write!(f, "{}", value),
            AssignLiteral(ref value) => write!(f, "{}", value),
            BinOp {
                ref op,
                ref left,
                ref right
            } => write!(f, "{} {} {}", op, left, right),
            Broadcast(ref child) => write!(f, "broadcast({})", child),
            Serialize(ref child) => write!(f, "serialize({})", child),
            Deserialize(ref child) => write!(f, "deserialize({})", child),
            Cast(ref child, ref ty) => write!(f, "cast({}, {})", child, ty),
            CUDF {
                ref symbol_name,
                ref args,
            } => {
                write!(f,
                       "cudf[{}]{}",
                       symbol_name,
                       join("(", ", ", ")", args.iter().map(|e| format!("{}", e))))
            }
            GetField {
                ref value,
                index,
            } => write!(f, "{}.${}", value, index),
            KeyExists {
                ref child,
                ref key,
            } => write!(f, "keyexists({}, {})", child, key),
            Length(ref child) => write!(f, "len({})", child),
            MakeStruct(ref elems) => {
                write!(f,
                       "{}",
                       join("{", ",", "}", elems.iter().map(|e| format!("{}", e))))
            }
            MakeVector(ref elems) => {
                write!(f,
                       "{}",
                       join("[", ", ", "]", elems.iter().map(|e| format!("{}", e))))
            }
            Merge {
                ref builder,
                ref value,
            } => write!(f, "merge({}, {})", builder, value),
            Negate(ref child) => write!(f, "-{}", child),
            Not(ref child) => write!(f, "!{}", child),
            NewBuilder {
                ref arg,
                ref ty,
            } => {
                let arg_str = if let Some(ref a) = *arg {
                    a.to_string()
                } else {
                    "".to_string()
                };
                write!(f, "new {}({})", ty, arg_str)
            }
            Lookup {
                ref child,
                ref index,
            } => write!(f, "lookup({}, {})", child, index),
            OptLookup {
                ref child,
                ref index,
            } => write!(f, "optlookup({}, {})", child, index),
            ParallelFor(ref pf) => {
                write!(f, "for [")?;
                for iter in &pf.data {
                    write!(f, "{}, ", iter)?;
                }
                write!(f, "] ")?;
                write!(f,
                       "{} {} {} {} F{} {}",
                       pf.builder,
                       pf.builder_arg,
                       pf.idx_arg,
                       pf.data_arg,
                       pf.body,
                       pf.innermost)?;
                Ok(())
            }
            Res(ref builder) => write!(f, "result({})", builder),
            Select {
                ref cond,
                ref on_true,
                ref on_false,
            } => write!(f, "select({}, {}, {})", cond, on_true, on_false),
            Slice {
                ref child,
                ref index,
                ref size,
            } => write!(f, "slice({}, {}, {})", child, index, size),
            Sort{ ref child, .. } => write!(f, "sort({})", child),
            ToVec(ref child) => write!(f, "toVec({})", child),
            UnaryOp {
                ref op,
                ref child
            } => write!(f, "{}({})", op, child),
        }
    }
}

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(ref sym) = self.output {
            write!(f, "{} = {}", sym, self.kind)
        } else {
            write!(f, "{}", self.kind)
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
            JumpBlock(block) => write!(f, "jump B{}", block),
            ProgramReturn(ref sym) => write!(f, "return {}", sym),
            EndFunction(ref sym) => write!(f, "end {}", sym),
            Crash => write!(f, "crash"),
        }
    }
}

impl fmt::Display for ParallelForIter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let iterkind = match self.kind {
            IterKind::ScalarIter => "iter",
            IterKind::SimdIter => "simditer",
            IterKind::FringeIter => "fringeiter",
            IterKind::NdIter => "nditer",
            IterKind::RangeIter => "rangeiter",
        };

        if self.shape.is_some() {
            /* NdIter. Note: end or stride aren't important here, so skpping those.
             * */
            write!(f,
                   "{}({}, {}, {}, {})",
                   iterkind,
                   self.data,
                   self.start.clone().unwrap(),
                   self.shape.clone().unwrap(),
                   self.strides.clone().unwrap())
        } else if self.start.is_some() {
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
        let loopbody = if self.loop_body {
            " (loopbody)"
        } else {
            ""
        };
        write!(f, "F{} -> {}{}:\n", self.id, &self.return_type, loopbody)?;
        write!(f, "Params:\n")?;
        let params_sorted: BTreeMap<&Symbol, &Type> = self.params.iter().collect();
        for (name, ty) in params_sorted {
            write!(f, "  {}: {}\n", name, ty)?;
        }
        write!(f, "Locals:\n")?;
        let locals_sorted: BTreeMap<&Symbol, &Type> = self.locals.iter().collect();
        for (name, ty) in locals_sorted {
            write!(f, "  {}: {}\n", name, ty)?;
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
            vars.extend(statement.kind.children().cloned());
        }
        vars.extend(block.terminator.children().cloned());
        for var in &vars {
            if prog.funcs[func_id].locals.get(&var) == None {
                prog.funcs[func_id]
                    .params
                    .insert(var.clone(), env.get(&var).unwrap().clone());
                closure.insert(var.clone());
            }
        }
        let mut inner_closure = HashSet::new();

        // Recurse into other called functions.
        for statement in &block.statements {
            use self::StatementKind::ParallelFor;
            match statement.kind {
                ParallelFor(ref pf) => {
                    sir_param_correction_helper(prog, pf.body, env, &mut inner_closure, visited);
                }
                _ => (),
            }
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

/// Recursive helper for `assign_return_types`.
///
/// This function recurses assigns a return type to the given function ID. It recurses into the
/// terminators and propogates the return types of the terminator to current function.
fn assign_return_types_helper(prog: &mut SirProgram, func: FunctionId) -> WeldResult<Type> {
    use sir::Terminator::*;

    // Base case - already visited this function!
    if prog.funcs[func].return_type != Unknown {
        return Ok(prog.funcs[func].return_type.clone())
    }

    // Symbol returned by the terminator.
    let mut return_symbol = None;
    {
        let ref function = prog.funcs[func];
        for block in function.blocks.iter() {
            match block.terminator {
                Branch { .. } => (),
                JumpBlock(_) => (),
                ProgramReturn(ref sym) | EndFunction(ref sym) => {
                    // Type should be set during AST -> SIR.
                    return_symbol = Some(sym.clone());
                },
                Crash => (),
            }
        }
    }

    // Need to do this nonsense to circumvent borrow checker...
    if let Some(symbol) = return_symbol {
        let return_type = prog.funcs[func].symbol_type(&symbol)?.clone();
        prog.funcs[func].return_type = return_type.clone();
        Ok(return_type)
    } else {
        // Indicates that the function did not return...
        unreachable!()
    }
}

/// Assigns return types to each function in the program.
fn assign_return_types(prog: &mut SirProgram) -> WeldResult<()> {
    for funcs in 0..prog.funcs.len() {
        assign_return_types_helper(prog, funcs)?;
    }
    Ok(())
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
            compile_err!("Unbound symbol {}", name.to_string())?;
        }
    }
    Ok(())
}

/// Convert an AST to a SIR program. Symbols must be unique in expr.
pub fn ast_to_sir(expr: &Expr) -> WeldResult<SirProgram> {
    if let ExprKind::Lambda { ref params, ref body } = expr.kind {
        let mut prog = SirProgram::new(&body.ty, params);
        prog.sym_gen = SymbolGenerator::from_expression(expr);
        for tp in params {
            prog.funcs[0].params.insert(tp.name.clone(), tp.ty.clone());
        }
        let first_block = prog.funcs[0].add_block();
        let (res_func, res_block, res_sym) = gen_expr(body, &mut prog, 0, first_block, &mut StatementTracker::new())?;
        prog.funcs[res_func].blocks[res_block].terminator = Terminator::ProgramReturn(res_sym);
        sir_param_correction(&mut prog)?;
        // second call is necessary in the case where there are loops in the call graph, since
        // some parameter dependencies may not have been propagated through back edges
        sir_param_correction(&mut prog)?;
        assign_return_types(&mut prog)?;
        Ok(prog)
    } else {
        compile_err!("Expression passed to ast_to_sir was not a Lambda")
    }
}

/// Helper method for gen_expr. Used to process the fields of ParallelForIter, like "start",
/// "shape" etc. Returns None, or the Symbol associated with the field. It also resets values for
/// cur_func, and cur_block.
fn get_iter_sym(opt : &Option<Box<Expr>>,
            prog: &mut SirProgram,
            cur_func: &mut FunctionId,
            cur_block: &mut BasicBlockId,
            tracker: &mut StatementTracker,
            body_func: FunctionId) -> WeldResult<Option<Symbol>> {
    if let &Some(ref opt_expr) = opt {
        let opt_res = gen_expr(&opt_expr, prog, *cur_func, *cur_block, tracker)?;
        /* TODO pari: Originally, in gen_expr cur_func, and cur_block were also being set - but this
        does not seem to have any effect. Could potentially remove this if it wasn't needed? All
        the tests seem to pass fine without it as well.
        */
        *cur_func = opt_res.0;
        *cur_block = opt_res.1;
        prog.funcs[body_func]
            .params
            .insert(opt_res.2.clone(), opt_expr.ty.clone());
        return Ok(Some(opt_res.2));
    } else {
        return Ok(None);
    };
}

/// Generate code to compute the expression `expr` starting at the current tail of `cur_block`,
/// possibly creating new basic blocks and functions in the process. Return the function and
/// basic block that the expression will be ready in, and its symbol therein.
fn gen_expr(expr: &Expr,
            prog: &mut SirProgram,
            cur_func: FunctionId,
            cur_block: BasicBlockId,
            tracker: &mut StatementTracker)
            -> WeldResult<(FunctionId, BasicBlockId, Symbol)> {
    use self::StatementKind::*;
    use self::Terminator::*;

    /*
    if prog.funcs[cur_func].return_type == Unknown {
        prog.funcs[cur_func].return_type = expr.ty.clone();
        debug!("F{} generating code for top expression\n{}", cur_func, expr.pretty_print());
    } else {
        trace!("F{} generating code for child expression\n{}", cur_func, expr.pretty_print());
    }
    */

    match expr.kind {
        ExprKind::Ident(ref sym) => Ok((cur_func, cur_block, sym.clone())),

        ExprKind::Literal(ref lit) => {
            let kind = AssignLiteral(lit.clone());
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Let {
            ref name,
            ref value,
            ref body,
        } => {
            let (cur_func, cur_block, val_sym) = gen_expr(value, prog, cur_func, cur_block, tracker)?;

            let kind = Assign(val_sym);
            tracker.named_symbol_for_statement(prog, cur_func, cur_block, &value.ty, kind, name.clone());

            let (cur_func, cur_block, res_sym) = gen_expr(body, prog, cur_func, cur_block, tracker)?;
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::BinOp {
            kind,
            ref left,
            ref right,
        } => {
            let (cur_func, cur_block, left_sym) = gen_expr(left, prog, cur_func, cur_block, tracker)?;
            let (cur_func, cur_block, right_sym) = gen_expr(right, prog, cur_func, cur_block, tracker)?;
            let kind = BinOp {
                op: kind,
                left: left_sym,
                right: right_sym,
            };
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::UnaryOp {
            kind,
            ref value,
        } => {
            let (cur_func, cur_block, value_sym) = gen_expr(value, prog, cur_func, cur_block, tracker)?;
            let kind = UnaryOp {
                op: kind,
                child: value_sym,
            };
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Negate(ref child_expr) => {
            let (cur_func, cur_block, child_sym) = gen_expr(child_expr, prog, cur_func, cur_block, tracker)?;
            let kind = Negate(child_sym);
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Not(ref child_expr) => {
            let (cur_func, cur_block, child_sym) = gen_expr(child_expr, prog, cur_func, cur_block, tracker)?;
            let kind = Not(child_sym);
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Broadcast(ref child_expr) => {
            let (cur_func, cur_block, child_sym) = gen_expr(child_expr, prog, cur_func, cur_block, tracker)?;
            let kind = Broadcast(child_sym);
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Serialize(ref child_expr) => {
            let (cur_func, cur_block, child_sym) = gen_expr(child_expr, prog, cur_func, cur_block, tracker)?;
            let kind = Serialize(child_sym);
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Deserialize {ref value, .. } => {
            let (cur_func, cur_block, child_sym) = gen_expr(value, prog, cur_func, cur_block, tracker)?;
            let kind = Deserialize(child_sym);
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Cast {ref child_expr, .. } => {
            let (cur_func, cur_block, child_sym) = gen_expr(child_expr, prog, cur_func, cur_block, tracker)?;
            let kind = Cast(child_sym, expr.ty.clone());
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Lookup {
            ref data,
            ref index,
        } => {
            let (cur_func, cur_block, data_sym) = gen_expr(data, prog, cur_func, cur_block, tracker)?;
            let (cur_func, cur_block, index_sym) = gen_expr(index, prog, cur_func, cur_block, tracker)?;

            let kind = Lookup {
                child: data_sym,
                index: index_sym.clone(),
            };
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::OptLookup {
            ref data,
            ref index,
        } => {
            let (cur_func, cur_block, data_sym) = gen_expr(data, prog, cur_func, cur_block, tracker)?;
            let (cur_func, cur_block, index_sym) = gen_expr(index, prog, cur_func, cur_block, tracker)?;

            let kind = OptLookup {
                child: data_sym,
                index: index_sym.clone(),
            };
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::KeyExists { ref data, ref key } => {
            let (cur_func, cur_block, data_sym) = gen_expr(data, prog, cur_func, cur_block, tracker)?;
            let (cur_func, cur_block, key_sym) = gen_expr(key, prog, cur_func, cur_block, tracker)?;
            let kind = KeyExists {
                child: data_sym,
                key: key_sym.clone(),
            };
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Slice {
            ref data,
            ref index,
            ref size,
        } => {
            let (cur_func, cur_block, data_sym) = gen_expr(data, prog, cur_func, cur_block, tracker)?;
            let (cur_func, cur_block, index_sym) = gen_expr(index, prog, cur_func, cur_block, tracker)?;
            let (cur_func, cur_block, size_sym) = gen_expr(size, prog, cur_func, cur_block, tracker)?;
            let kind = Slice {
                child: data_sym,
                index: index_sym.clone(),
                size: size_sym.clone(),
            };
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Sort {
            ref data,
            ref keyfunc,
        } => {
            if let ExprKind::Lambda {
                       ref params,
                       ref body,
            } = keyfunc.kind {
                let keyfunc_id = prog.add_func();
                let keyblock = prog.funcs[keyfunc_id].add_block();
                let (keyfunc_id, keyblock, key_sym) = gen_expr(body, prog, keyfunc_id, keyblock, tracker)?;

                prog.funcs[keyfunc_id].params.insert(params[0].name.clone(), params[0].ty.clone());
                prog.funcs[keyfunc_id].blocks[keyblock].terminator = Terminator::ProgramReturn(key_sym.clone());

                let (cur_func, cur_block, data_sym) = gen_expr(data, prog, cur_func, cur_block, tracker)?;
                let key_function = prog.funcs[keyfunc_id].clone();

                let kind = Sort {
                    child: data_sym,
                    keyfunc: key_function
                };
                let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
                Ok((cur_func, cur_block, res_sym))
            } else {
                compile_err!("Sort key function expected lambda type, instead {:?} provided", keyfunc.ty)
            }
        }
        ExprKind::Select {
            ref cond,
            ref on_true,
            ref on_false,
        } => {
            let (cur_func, cur_block, cond_sym) = gen_expr(cond, prog, cur_func, cur_block, tracker)?;
            let (cur_func, cur_block, true_sym) = gen_expr(on_true, prog, cur_func, cur_block, tracker)?;
            let (cur_func, cur_block, false_sym) = gen_expr(on_false, prog, cur_func, cur_block, tracker)?;
            let kind = Select {
                cond: cond_sym,
                on_true: true_sym.clone(),
                on_false: false_sym.clone(),
            };
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::ToVec { ref child_expr } => {
            let (cur_func, cur_block, child_sym) = gen_expr(child_expr, prog, cur_func, cur_block, tracker)?;
            let kind = ToVec(child_sym);
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::Length { ref data } => {
            let (cur_func, cur_block, child_sym) = gen_expr(data, prog, cur_func, cur_block, tracker)?;
            let kind = Length(child_sym);
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::If {
            ref cond,
            ref on_true,
            ref on_false,
        } => {
            let (cur_func, cur_block, cond_sym) = gen_expr(cond, prog, cur_func, cur_block, tracker)?;
            let true_block = prog.funcs[cur_func].add_block();
            let false_block = prog.funcs[cur_func].add_block();
            prog.funcs[cur_func].blocks[cur_block].terminator = Branch {
                cond: cond_sym,
                on_true: true_block,
                on_false: false_block,
            };
            let (true_func, true_block, true_sym) = gen_expr(on_true, prog, cur_func, true_block, tracker)?;
            let (false_func, false_block, false_sym) = gen_expr(on_false, prog, cur_func, false_block, tracker)?;
            let res_sym = prog.add_local(&expr.ty, true_func);
            prog.funcs[true_func].blocks[true_block].add_statement(Statement::new(Some(res_sym.clone()), Assign(true_sym)));
            prog.funcs[false_func].blocks[false_block].add_statement(Statement::new(Some(res_sym.clone()), Assign(false_sym)));

            let cont_block = prog.funcs[cur_func].add_block();
            prog.funcs[true_func].blocks[true_block].terminator = JumpBlock(cont_block);
            prog.funcs[false_func].blocks[false_block].terminator = JumpBlock(cont_block);
            Ok((cur_func, cont_block, res_sym))
        }

        ExprKind::Iterate {
            ref initial,
            ref update_func,
        } => {
            // Generate the intial value.
            let (cur_func, cur_block, initial_sym) = gen_expr(initial, prog, cur_func, cur_block, tracker)?;

            // Pull out the argument name and function body and validate that things type-check.
            let argument_sym;
            let func_body;
            match update_func.kind {
                ExprKind::Lambda { ref params, ref body } if params.len() == 1 => {
                    argument_sym = &params[0].name;
                    func_body = body;
                    if params[0].ty != initial.ty {
                        return compile_err!("Wrong argument type for body of Iterate");
                    }
                    if func_body.ty != Struct(vec![initial.ty.clone(), Scalar(ScalarKind::Bool)]) {
                        return compile_err!("Wrong return type for body of Iterate");
                    }
                    prog.add_local_named(&params[0].ty, argument_sym, cur_func);
                }
                _ => return compile_err!("Argument of Iterate was not a Lambda")
            }

            prog.funcs[cur_func].blocks[cur_block].add_statement(Statement::new(Some(argument_sym.clone()), Assign(initial_sym)));

            let body_start_block = prog.funcs[cur_func].add_block();

            // Jump to where the body starts
            prog.funcs[cur_func].blocks[cur_block].terminator = JumpBlock(body_start_block);

            // Generate the loop's body, which will work on argument_sym and produce result_sym.
            // The type of result_sym will be {ArgType, bool} and we will repeat the body if the bool is true.
            let (body_end_func, body_end_block, result_sym) =
                gen_expr(func_body, prog, cur_func, body_start_block, tracker)?;

            // After the body, unpack the {state, bool} struct into symbols argument_sym and continue_sym.
            let continue_sym = prog.add_local(&Scalar(ScalarKind::Bool), body_end_func);

            prog.funcs[body_end_func].blocks[body_end_block].add_statement(
                Statement::new(Some(argument_sym.clone()), GetField { value: result_sym.clone(), index: 0 }));
            prog.funcs[body_end_func].blocks[body_end_block].add_statement(
                Statement::new(Some(continue_sym.clone()), GetField { value: result_sym.clone(), index: 1 }));

            // Create two more blocks so we can branch on continue_sym
            let repeat_block = prog.funcs[body_end_func].add_block();
            let finish_block = prog.funcs[body_end_func].add_block();
            prog.funcs[body_end_func].blocks[body_end_block].terminator =
                Branch { cond: continue_sym, on_true: repeat_block, on_false: finish_block };

            // Do a normal JumpBlock since it should be in the same function.
            assert!(body_end_func == cur_func && cur_func == cur_func);
            prog.funcs[body_end_func].blocks[repeat_block].terminator = JumpBlock(body_start_block);

            // Our final value is available in finish_block.
            Ok((body_end_func, finish_block, argument_sym.clone()))
        }

        ExprKind::Merge {
            ref builder,
            ref value,
        } => {
            // This expression doesn't return a symbol, so just add a statement for it directly
            // instead of calling the tracker.
            let (cur_func, cur_block, builder_sym) = gen_expr(builder, prog, cur_func, cur_block, tracker)?;
            let (cur_func, cur_block, elem_sym) = gen_expr(value, prog, cur_func, cur_block, tracker)?;
            prog.funcs[cur_func].blocks[cur_block].add_statement(Statement::new(None, Merge {
                                                                     builder: builder_sym.clone(),
                                                                     value: elem_sym,
                                                                 }));
            Ok((cur_func, cur_block, builder_sym))
        }

        ExprKind::Res { ref builder } => {
            let (cur_func, cur_block, builder_sym) = gen_expr(builder, prog, cur_func, cur_block, tracker)?;
            let kind = Res(builder_sym);
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::NewBuilder(ref arg) => {
            let (cur_func, cur_block, arg_sym) = if let Some(ref a) = *arg {
                let (cur_func, cur_block, arg_sym) = gen_expr(a, prog, cur_func, cur_block, tracker)?;
                (cur_func, cur_block, Some(arg_sym))
            } else {
                (cur_func, cur_block, None)
            };

            // NewBuilder is special, since they are stateful objects - we can't alias them.
            let res_sym = prog.add_local(&expr.ty, cur_func);
            prog.funcs[cur_func].blocks[cur_block].add_statement(Statement::new(Some(res_sym.clone()), NewBuilder {
                                                                     arg: arg_sym,
                                                                     ty: expr.ty.clone(),
                                                                 }));
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::MakeStruct { ref elems } => {
            let mut syms = vec![];
            let (mut cur_func, mut cur_block, mut sym) =
                gen_expr(&elems[0], prog, cur_func, cur_block, tracker)?;
            syms.push(sym);
            for elem in elems.iter().skip(1) {
                let r = gen_expr(elem, prog, cur_func, cur_block, tracker)?;
                cur_func = r.0;
                cur_block = r.1;
                sym = r.2;
                syms.push(sym);
            }
            let kind = MakeStruct(syms);
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::MakeVector { ref elems } => {
            let mut syms = vec![];
            let mut cur_func = cur_func;
            let mut cur_block = cur_block;
            for elem in elems.iter() {
                let r = gen_expr(elem, prog, cur_func, cur_block, tracker)?;
                cur_func = r.0;
                cur_block = r.1;
                let sym = r.2;
                syms.push(sym);
            }
            let kind = MakeVector(syms);
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
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
                let r = gen_expr(arg, prog, cur_func, cur_block, tracker)?;
                cur_func = r.0;
                cur_block = r.1;
                let sym = r.2;
                syms.push(sym);
            }
            let kind = CUDF {
                args: syms,
                symbol_name: sym_name.clone(),
            };
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &expr.ty, kind);
            Ok((cur_func, cur_block, res_sym))
        }

        ExprKind::GetField { ref expr, index } => {
            let (cur_func, cur_block, struct_sym) = gen_expr(expr, prog, cur_func, cur_block, tracker)?;
            let field_ty = match expr.ty {
                super::ast::Type::Struct(ref v) => &v[index as usize],
                _ => {
                    compile_err!("Internal error: tried to get field of type {}",
                              &expr.ty)?
                }
            };

            let kind = GetField {
                value: struct_sym,
                index: index,
            };
            let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &field_ty, kind);
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
                    gen_expr(builder, prog, cur_func, cur_block, tracker)?;
                let body_func = prog.add_func();
                prog.funcs[body_func].loop_body = true;
                let body_block = prog.funcs[body_func].add_block();
                prog.add_local_named(&params[0].ty, &params[0].name, body_func);
                prog.add_local_named(&params[1].ty, &params[1].name, body_func);
                prog.add_local_named(&params[2].ty, &params[2].name, body_func);

                // Register the data, index, and builder arguments are loop variables.
                prog.add_loop_variable(&params[0].name, body_func);
                prog.add_loop_variable(&params[1].name, body_func);
                prog.add_loop_variable(&params[2].name, body_func);

                prog.funcs[body_func]
                    .params
                    .insert(builder_sym.clone(), builder.ty.clone());
                let mut cur_func = cur_func;
                let mut cur_block = cur_block;
                let mut pf_iters: Vec<ParallelForIter> = Vec::new();
                for iter in iters.iter() {
                    let data_res = gen_expr(&iter.data, prog, cur_func, cur_block, tracker)?;
                    cur_func = data_res.0;
                    cur_block = data_res.1;
                    prog.funcs[body_func]
                        .params
                        .insert(data_res.2.clone(), iter.data.ty.clone());
                    let start_sym = try!(get_iter_sym(&iter.start, prog, &mut cur_func, &mut cur_block,
                                                      tracker, body_func));
                    let end_sym = try!(get_iter_sym(&iter.end, prog, &mut cur_func, &mut cur_block,
                                                    tracker, body_func));
                    let stride_sym = try!(get_iter_sym(&iter.stride, prog, &mut cur_func, &mut cur_block,
                                                       tracker, body_func));
                    let shape_sym = try!(get_iter_sym(&iter.shape, prog, &mut cur_func, &mut cur_block,
                                                       tracker, body_func));
                    let strides_sym = try!(get_iter_sym(&iter.strides, prog, &mut cur_func, &mut cur_block,
                                                        tracker, body_func));
                    pf_iters.push(ParallelForIter {
                                      data: data_res.2,
                                      start: start_sym,
                                      end: end_sym,
                                      stride: stride_sym,
                                      kind: iter.kind.clone(),
                                      shape: shape_sym,
                                      strides: strides_sym,
                                  });
                }
                let (body_end_func, body_end_block, result_sym) =
                    gen_expr(body, prog, body_func, body_block, tracker)?;
                prog.funcs[body_end_func].blocks[body_end_block].terminator = EndFunction(result_sym);

                // Check whether the loop is the innermost one.
                let mut is_innermost = true;
                body.traverse(&mut |ref e| if let ExprKind::For { .. } = e.kind {
                                       is_innermost = false;
                                   });

                let kind = ParallelFor(ParallelForData {
                                    data: pf_iters,
                                    builder: builder_sym.clone(),
                                    builder_arg: params[0].name.clone(),
                                    idx_arg: params[1].name.clone(),
                                    data_arg: params[2].name.clone(),
                                    body: body_func,
                                    innermost: is_innermost,
                                });

                let res_sym = tracker.symbol_for_statement(prog, cur_func, cur_block, &builder.ty, kind);
                Ok((cur_func, cur_block, res_sym))
            } else {
                compile_err!("Argument to For was not a Lambda: {}", func.pretty_print())
            }
        }

        _ => compile_err!("Unsupported expression: {}", expr.pretty_print())
    }
}
