use std::collections::HashMap;

use llvm;
use llvm::{Compile, ExecutionEngine};
use llvm::value::Predicate;

use super::ast::*;
use super::ast::Type::*;
use super::ast::ExprKind::*;
use super::ast::ScalarKind::*;
use super::error::*;
use super::program::Program;
use super::pretty_print::*;
use super::type_inference;
use super::macro_processor;

// TODO(wcrichto): add prelude to generator
static PRELUDE_CODE: &'static str = include_str!("resources/prelude.ll");

#[macro_export]
macro_rules! make_generator {
    ($generator:ident) => {
        let ctx = ::llvm::Context::new();
        let module = llvm::Module::new("dummy", &ctx);
        let engine = {
            use llvm::ExecutionEngine;
            ::llvm::JitEngine::new(&module, ::llvm::JitOptions {opt_level: 0})
                .expect("Jit not initialized")
        };
        let mut $generator = Generator::new(&ctx, &module, &engine);
    }
}

struct SymbolStack<'a> {
    symbols: HashMap<Symbol, Vec<&'a llvm::Value>>
}

impl<'a> SymbolStack<'a> {
    pub fn new() -> SymbolStack<'a> {
        SymbolStack {
            symbols: HashMap::new()
        }
    }

    pub fn push(&mut self, key: &Symbol, val: &'a llvm::Value) {
        let mut stack = self.symbols.entry(key.clone()).or_insert(Vec::new());
        stack.push(val);
    }

    pub fn peek(&self, key: &Symbol) -> WeldResult<&'a llvm::Value> {
        if !self.symbols.contains_key(key) {
            weld_err!("Missing symbol {:?}", key)
        } else {
            Ok(self.symbols[key].last().unwrap())
        }
    }

    pub fn pop(&mut self, key: &Symbol) -> WeldResult<()> {
        let mut stack = match self.symbols.get_mut(key) {
            Some(stack) => stack,
            None => return weld_err!("Missing symbol {:?}", key)
        };
        stack.pop();
        Ok(())
    }
}

/// Struct used to track state while generating a function.
struct FunctionContext<'a> {
    pub builder: &'a llvm::CSemiBox<'a, llvm::Builder>,
    pub symbols: SymbolStack<'a>
}

impl<'a> FunctionContext<'a> {
    pub fn new(builder: &'a llvm::CSemiBox<'a, llvm::Builder>) -> FunctionContext<'a> {
        FunctionContext {
            builder: builder,
            symbols: SymbolStack::new()
        }
    }
}

/// Generates LLVM code for one or more modules.
pub struct Generator<'a> {
    ctx: &'a llvm::Context,
    module: &'a llvm::CSemiBox<'a, llvm::Module>,
    engine: &'a llvm::CSemiBox<'a, llvm::JitEngine>
}

impl<'a> Drop for Generator<'a> {
    fn drop(&mut self) {
        self.engine.remove_module(self.module);
    }
}

impl<'a> Generator<'a> {
    pub fn new(
        ctx: &'a llvm::Context,
        module: &'a llvm::CSemiBox<'a, llvm::Module>,
        engine: &'a llvm::CSemiBox<'a, llvm::JitEngine>)
        -> Generator<'a>
    {
        Generator {
            ctx: ctx,
            engine: engine,
            module: module
        }
    }

    /// Return all the code generated so far.
    pub fn result(&self) -> String {
        format!("{:?}", self.module)
    }

    pub fn get_function<I, O>(&self, name: String) -> WeldResult<Box<extern fn(I) -> O>> {
        let f = match self.engine.find_function(&name) {
            Some(f) => f,
            None => { return weld_err!("Function {} is missing", name); }
        };
        Ok(Box::new(unsafe { self.engine.get_function(f) }))
    }

    /// Generate a compiled LLVM module from a program whose body is a function.
    pub fn add_program(&mut self, program: &Program, name: &str) -> WeldResult<()> {
        let mut expr = macro_processor::process_program(program)?;
        type_inference::infer_types(&mut expr)?;
        let expr = expr.to_typed()?;
        match expr.kind {
            Lambda(ref params, ref body) => {
                self.add_function_on_pointers(&self.module, name, params, body)?;
                Ok(())
            },
            _ => weld_err!("Expression passed to compile_function must be a Lambda")
        }
    }

    /// Add a function to the generated program.
    pub fn add_function(
        &self,
        module: &'a llvm::CSemiBox<'a, llvm::Module>,
        name: &str,
        args: &Vec<TypedParameter>,
        body: &TypedExpr)
        -> WeldResult<&'a llvm::Function>
    {
        let res_type = self.llvm_type(&body.ty)?;
        let arg_types = {
            let mut ts = Vec::new();
            for arg in args {
                ts.push(self.llvm_type(&arg.ty)?);
            }
            ts
        };
        let func_type = llvm::FunctionType::new(res_type, &arg_types);
        let func = module.add_function(name, func_type);
        let entry = func.append("entry");

        let builder = llvm::Builder::new(self.ctx);
        builder.position_at_end(entry);
        let mut fn_ctx = FunctionContext::new(&builder);
        for (i, arg) in args.iter().enumerate() {
            fn_ctx.symbols.push(&arg.name, &func[i]);
        }
        let ret = self.gen_expr(&body, &mut fn_ctx)?;
        builder.build_ret(ret);

        module.verify().expect("Module verification failed");

        Ok(func)
    }

    /// Add a function to the generated program, passing its parameters and return value through
    /// pointers encoded as i64. This is used for the main entry point function into Weld modules
    /// to pass them arbitrary structures.
    pub fn add_function_on_pointers(
        &self,
        module: &'a llvm::CSemiBox<'a, llvm::Module>,
        name: &str,
        args: &Vec<TypedParameter>,
        body: &TypedExpr)
        -> WeldResult<()>
    {
        // First add the function on raw values, which we'll call from the pointer version.
        let raw_function_name = format!("{}.raw", name);
        let raw_function = self.add_function(&module, &raw_function_name, args, body)?;

        // Define a struct with all the argument types as fields
        let args_struct = Struct(args.iter().map(|a| a.ty.clone()).collect());
        let args_type = self.llvm_type(&args_struct)?;
        let args_ptr_type = llvm::PointerType::new(args_type);
        let res_type = self.llvm_type(&body.ty)?;

        let ty_i64 = llvm::Type::get::<i64>(self.ctx);
        let func_type = llvm::FunctionType::new(ty_i64, &[ty_i64]);
        let func = module.add_function(name, func_type);
        let entry = func.append("entry");

        let builder = llvm::Builder::new(self.ctx);
        builder.position_at_end(entry);

        let arg = &func[0];
        let args_typed = builder.build_int_to_ptr(arg, args_ptr_type);
        let args_val = builder.build_load(args_typed);
        let arg_list = (0..args.len()).into_iter().map(|i| {
            builder.build_extract_value(args_val, i)
        }).collect::<Vec<&llvm::Value>>();

        let res_slot = builder.build_alloca(res_type);
        let res_val = builder.build_call(raw_function, &arg_list);
        builder.build_store(res_val, res_slot);
        let res_address = builder.build_ptr_to_int(res_slot, ty_i64);
        builder.build_ret(res_address);

        module.verify().expect("Module verification failed");

        Ok(())
    }

    /// Return the LLVM type name corresponding to a Weld type.
    fn llvm_type(&self, ty: &Type) -> WeldResult<&'a llvm::Type> {
        match *ty {
            Scalar(Bool) => Ok(llvm::Type::get::<bool>(self.ctx)),
            Scalar(I32) => Ok(llvm::Type::get::<i32>(self.ctx)),
            Scalar(I64) => Ok(llvm::Type::get::<i64>(self.ctx)),
            Scalar(F32) => Ok(llvm::Type::get::<f32>(self.ctx)),
            Scalar(F64) => Ok(llvm::Type::get::<f64>(self.ctx)),

            Struct(ref fields) => {
                let mut field_types = Vec::new();
                for field in fields {
                    field_types.push(self.llvm_type(field)?);
                }
                Ok(llvm::StructType::new(self.ctx, &field_types, true))
            }

            Vector(ref elem) => {
                // TODO(wcrichto): where is vector type length?
                Ok(llvm::VectorType::new(self.llvm_type(elem)?, 4))
            }

            _ => weld_err!("Unsupported type {}", print_type(ty))
        }
    }

    fn predicate_for_binop(&self, op_kind: BinOpKind) -> Predicate {
        match op_kind {
            BinOpKind::Equal => Predicate::Equal,
            BinOpKind::NotEqual => Predicate::NotEqual,
            BinOpKind::GreaterThan => Predicate::GreaterThan,
            BinOpKind::GreaterThanOrEqual => Predicate::GreaterThanOrEqual,
            BinOpKind::LessThan => Predicate::LessThan,
            BinOpKind::LessThanOrEqual => Predicate::LessThanOrEqual,
            _ => unreachable!()
        }
    }

    /// Return the name of the LLVM instruction for a binary operation on a specific type.
    fn llvm_binop(
        &self,
        op_kind: BinOpKind,
        left: &'a llvm::Value,
        right: &'a llvm::Value,
        builder: &'a llvm::CSemiBox<'a, llvm::Builder>)
        -> WeldResult<&'a llvm::Value>
    {
        match op_kind {
            BinOpKind::Add => Ok(builder.build_add(left, right)),
            BinOpKind::Subtract => Ok(builder.build_sub(left, right)),
            BinOpKind::Multiply => Ok(builder.build_mul(left, right)),
            BinOpKind::Divide => Ok(builder.build_div(left, right)),
            BinOpKind::Equal
                | BinOpKind::NotEqual
                | BinOpKind::GreaterThan
                | BinOpKind::GreaterThanOrEqual
                | BinOpKind::LessThan
                | BinOpKind::LessThanOrEqual =>
                Ok(builder.build_cmp(left, right, self.predicate_for_binop(op_kind))),
            _ => weld_err!("Unsupported binary op: {}", op_kind)
        }
    }

    /// Add an expression to a CodeBuilder, possibly generating prelude code earlier, and return
    /// a string that can be used to represent its result later (e.g. %var if introducing a local
    /// variable or an integer constant otherwise).
    fn gen_expr(
        &self,
        expr: &TypedExpr,
        fn_ctx: &mut FunctionContext<'a>,
    ) -> WeldResult<&'a llvm::Value> {
        match expr.kind {
            I32Literal(value) => Ok(value.compile(self.ctx)),
            I64Literal(value) => Ok(value.compile(self.ctx)),
            F32Literal(value) => Ok(value.compile(self.ctx)),
            F64Literal(value) => Ok(value.compile(self.ctx)),
            BoolLiteral(value) => Ok(value.compile(self.ctx)),

            Ident(ref symbol) => {
                Ok(fn_ctx.symbols.peek(symbol)?)
            },

            BinOp(kind, ref left, ref right) => {
                let left = self.gen_expr(left, fn_ctx)?;
                let right = self.gen_expr(right, fn_ctx)?;
                Ok(self.llvm_binop(kind, left, right, fn_ctx.builder)?)
            },

            Let(ref name, ref value, ref body) => {
                let value = self.gen_expr(value, fn_ctx)?;
                fn_ctx.symbols.push(name, value);
                let body = self.gen_expr(body, fn_ctx)?;
                fn_ctx.symbols.pop(name)?;
                Ok(body)
            },

            If(ref cond, ref on_true, ref on_false) => {
                let cond = self.gen_expr(cond, fn_ctx)?;
                let on_true = self.gen_expr(on_true, fn_ctx)?;
                let on_false = self.gen_expr(on_false, fn_ctx)?;
                Ok(fn_ctx.builder.build_select(cond, on_true, on_false))
            },

            _ => weld_err!("Unsupported expression: {}", print_expr(expr))
        }
    }
}

macro_rules! make_test {
    ($name:ident, $code:expr, $input:expr, $expected:expr) => {
        #[test]
        fn $name() {
            use super::parser::*;
            let code = $code;
            make_generator!(generator);
            generator.add_program(&parse_program(code).expect("Could not parse"), "run")
                .expect("Failed to add program");
            let f: Box<extern fn(i64) -> *const i32> =
                generator.get_function("run".into()).expect("No function");
            let input: i32 = $input;
            let result = f(&input as *const i32 as i64);
            let result = unsafe { *result };
            assert_eq!(result, $expected);
            // TODO: Free result
        }
    }
}

make_test!(basic_program, "|| 40+2", 0, 42);
make_test!(program_with_args, "|x:i32| 40+x", 2, 42);
make_test!(let_statement, "|x:i32| let y = 40+x; y+2", 2, 44);
make_test!(if_statement, "|x:i32| if(true, 3, 4)", 0, 3);
make_test!(comparison_false, "|x:i32| if(x>10, x, 10)", 2, 10);
make_test!(comparison_true, "|x:i32| if(x>10, x, 10)", 20, 20);
