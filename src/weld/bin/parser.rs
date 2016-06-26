extern crate weld;

use std::io::{stdin, stdout, Write};
use weld::grammar::*;
use weld::type_inference::infer_types;
use weld::pretty_print::*;

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
        if trimmed != "" {
            let expr = parse_expr(trimmed);
            match expr {
                Ok(mut expr) => {
                    println!("Raw structure:\n{:?}\n", expr);
                    println!("Pretty printed:\n{}\n", print_expr(&expr));
                    match infer_types(&mut expr) {
                        Ok(_) => println!("After type inference:\n{}\n\nExpression type: {}\n",
                            print_typed_expr(&expr), print_optional_type(&expr.ty)),
                        Err(e) => println!("Error during type inference: {}", e)
                    }
                }
                Err(e) => {
                    println!("Error during parsing: {:?}", e)
                }
            }
        }
    }
}