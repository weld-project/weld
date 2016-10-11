extern crate easy_ll;
extern crate weld;

use std::io::{stdin, stdout, Write};

use weld::ast::ExprKind::*;
use weld::llvm::LlvmGenerator;
use weld::macro_processor;
use weld::parser::*;
use weld::pretty_print::*;
use weld::transforms;
use weld::type_inference::*;

fn main() {
    loop {
        print!("> ");
        stdout().flush().unwrap();
        let mut line = String::new();
        stdin().read_line(&mut line).unwrap();
        if &line == "" {
            println!("");
            return  // Reached EOF
        }

        let trimmed = line.trim();
        if trimmed == "" {
            continue;
        }

        let program = parse_program(trimmed);
        if let Err(ref e) = program {
            println!("Error during parsing: {:?}", e);
            continue;
        }
        let program = program.unwrap();
        println!("Raw structure:\n{:?}\n", program);

        let expr = macro_processor::process_program(&program);
        if let Err(ref e) = expr {
            println!("Error during macro substitution: {}", e);
            continue;
        }
        let mut expr = expr.unwrap();
        println!("After macro substitution:\n{}\n", print_expr(&expr));

        if let Err(ref e) = transforms::inline_apply(&mut expr) {
            println!("Error during inlining applies: {}\n", e);
        }
        println!("After inlining applies:\n{}\n", print_expr(&expr));

        if let Err(ref e) = infer_types(&mut expr) {
            println!("Error during type inference: {}\n", e);
            println!("Partially inferred types:\n{}\n", print_typed_expr(&expr));
            continue;
        }
        println!("After type inference:\n{}\n", print_typed_expr(&expr));
        println!("Expression type: {}\n", print_type(&expr.ty));

        let expr = expr.to_typed().unwrap();
        if let Lambda(ref args, ref body) = expr.kind {
            let mut generator = LlvmGenerator::new();
            if let Err(ref e) = generator.add_function_on_pointers("run", args, body) {
                println!("Error during LLVM code gen:\n{}\n", e);
                continue;
            }
            let llvm_code = generator.result();
            println!("LLVM code:\n{}\n", llvm_code);

            if let Err(ref e) = easy_ll::compile_module(&llvm_code) {
                println!("Error during LLVM compilation:\n{}\n", e);
                continue;
            }
            println!("LLVM module compiled successfully\n")
        } else {
            println!("Expression is not a function, so not compiling to LLVM.\n")
        }
    }
}
