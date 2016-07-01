extern crate weld;

use std::io::{stdin, stdout, Write};
use weld::grammar::*;
use weld::pretty_print::*;
use weld::type_inference::*;
use weld::macro_processor;

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
        // TODO: don't have so much nested stuff here
        let trimmed = line.trim();
        if trimmed != "" {
            match parse_program(trimmed) {
                Ok(program) => {
                    println!("Raw structure:\n{:?}\n", program);
                    match macro_processor::process_program(&program) {
                        Ok(ref mut expr) => {
                            println!("After macro substitution:\n{}\n", print_expr(expr));
                            match infer_types(expr) {
                                Ok(_) => println!("After type inference:\n{}\n\nExpression type: {}\n",
                                    print_typed_expr(expr), print_type(&expr.ty)),
                                Err(e) => println!("Error during type inference: {}", e)
                            }
                        }
                        Err(e) => println!("Error during macro substitution: {}", e)
                    }
                }
                Err(e) => println!("Error during parsing: {:?}", e)
            }
        }
    }
}