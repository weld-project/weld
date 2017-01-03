#[macro_use]
extern crate weld;
extern crate llvm;
extern crate rustyline;

use rustyline::error::ReadlineError;
use rustyline::Editor;
use std::env;
use std::path::PathBuf;
use weld::codegen::Generator;
use weld::parser::*;

fn main() {
    let home_path = env::home_dir().unwrap_or(PathBuf::new());
    let history_file_path = home_path.join(".weld_history");
    let history_file_path = history_file_path.to_str().unwrap_or(".weld_history");

    let mut rl = Editor::<()>::new();
    rl.load_history(&history_file_path).unwrap();

    loop {
        let raw_readline = rl.readline(">> ");
        let readline;
        match raw_readline {
            Ok(raw_readline) => {
                rl.add_history_entry(&raw_readline);
                readline = raw_readline;
            },
            Err(ReadlineError::Interrupted) => {
                println!("Exiting!");
                break
            },
            Err(ReadlineError::Eof) => {
                println!("Exiting!");
                break
            },
            Err(err) => {
                println!("Error: {:?}", err);
                break
            }
        }

        let trimmed = readline.trim();
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

        make_generator!(generator);
        if let Err(ref e) = generator.add_program(&program) {
            println!("Error during LLVM code gen:\n{}\n", e);
            continue;
        }
        let llvm_code = generator.result();
        println!("LLVM code:\n{}\n", llvm_code);

        println!("LLVM module compiled successfully\n")

    }
    rl.save_history(&history_file_path).unwrap();
}
