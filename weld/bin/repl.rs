extern crate rustyline;
extern crate easy_ll;
extern crate weld;

extern crate libc;

use rustyline::error::ReadlineError;
use rustyline::Editor;
use std::env;
use std::path::Path;
use std::path::PathBuf;
use std::fs::File;
use std::error::Error;
use std::io::prelude::*;
use std::fmt;
use std::collections::HashMap;

use weld::*;
use weld::llvm::LlvmGenerator;
use weld::parser::*;
use weld::passes::*;
use weld::pretty_print::*;
use weld::type_inference::*;
use weld::sir::ast_to_sir;
use weld::util::load_runtime_library;
use weld::util::MERGER_BC;

use weld::transforms;

enum ReplCommands {
    LoadFile,
}

impl fmt::Display for ReplCommands {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ReplCommands::LoadFile => write!(f, "load"),
        }
    }
}

/// Processes the LoadFile command.
///
/// The argument is a filename containing a Weld program. Returns the string
/// representation of the program or an error with an error message.
fn process_loadfile(arg: String) -> Result<String, String> {
    if arg.len() == 0 {
        return Err("Error: expected argument for command 'load'".to_string());
    }
    let path = Path::new(&arg);
    let path_display = path.display();
    let mut file;
    match File::open(&path) {
        Err(why) => {
            return Err(format!("Error: couldn't open {}: {}",
                               path_display,
                               why.description()));
        }
        Ok(res) => {
            file = res;
        }
    }

    let mut contents = String::new();
    match file.read_to_string(&mut contents) {
        Err(why) => {
            return Err(format!("Error: couldn't read {}: {}",
                               path_display,
                               why.description()));
        }
        _ => {}
    }
    Ok(contents.trim().to_string())
}

fn main() {
    let home_path = env::home_dir().unwrap_or(PathBuf::new());
    let history_file_path = home_path.join(".weld_history");
    let history_file_path = history_file_path.to_str().unwrap_or(".weld_history");

    let mut reserved_words = HashMap::new();
    reserved_words.insert(ReplCommands::LoadFile.to_string(), ReplCommands::LoadFile);

    let mut rl = Editor::<()>::new();
    if let Err(_) = rl.load_history(&history_file_path) {}

    loop {
        let raw_readline = rl.readline(">> ");
        let readline;
        match raw_readline {
            Ok(raw_readline) => {
                rl.add_history_entry(&raw_readline);
                readline = raw_readline;
            }
            Err(ReadlineError::Interrupted) => {
                println!("Exiting!");
                break;
            }
            Err(ReadlineError::Eof) => {
                println!("Exiting!");
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }

        let trimmed = readline.trim();
        if trimmed == "" {
            continue;
        }

        let program;

        // Do some basic token parsing here.
        let mut tokens = trimmed.splitn(2, " ");
        let command = tokens.next().unwrap();
        let arg = tokens.next().unwrap_or("");
        if reserved_words.contains_key(command) {
            let command = reserved_words.get(command).unwrap();
            match *command {
                ReplCommands::LoadFile => {
                    match process_loadfile(arg.to_string()) {
                        Err(s) => {
                            println!("{}", s);
                            continue;
                        }
                        Ok(code) => {
                            program = parse_program(&code);
                        }
                    }
                }
            }
        } else {
            program = parse_program(trimmed);
        }

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

        if let Err(ref e) = transforms::uniquify(&mut expr) {
            println!("Error during uniquify: {}\n", e);
            continue;
        }
        println!("After uniquify:\n{}\n", print_expr(&expr));

        if let Err(ref e) = infer_types(&mut expr) {
            println!("Error during type inference: {}\n", e);
            println!("Partially inferred types:\n{}\n", print_typed_expr(&expr));
            continue;
        }
        println!("After type inference:\n{}\n", print_typed_expr(&expr));
        println!("Expression type: {}\n", print_type(&expr.ty));

        let mut expr = expr.to_typed().unwrap();

        let passes: Vec<&Pass> = vec![OPTIMIZATION_PASSES.get("inline-apply").unwrap(),
                                      OPTIMIZATION_PASSES.get("inline-let").unwrap(),
                                      OPTIMIZATION_PASSES.get("inline-zip").unwrap(),
                                      OPTIMIZATION_PASSES.get("loop-fusion").unwrap()];

        for i in 0..passes.len() {
            println!("Applying pass {}", passes[i].pass_name());
            passes[i].transform(&mut expr).unwrap();
            println!("After {} pass:\n{}\n",
                     passes[i].pass_name(),
                     print_expr(&expr));
        }

        if let Err(ref e) = transforms::uniquify(&mut expr) {
            println!("Error during uniquify: {}\n", e);
            continue;
        }

        println!("final program : {}", print_typed_expr(&expr));
        println!("final program raw: {:?}", expr);

        let sir_result = ast_to_sir(&expr);
        match sir_result {
            Ok(sir) => {
                println!("SIR representation:\n{}\n", &sir);
                let mut llvm_gen = LlvmGenerator::new();
                if let Err(ref e) = llvm_gen.add_function_on_pointers("run", &sir) {
                    println!("Error during LLVM code gen:\n{}\n", e);
                } else {
                    let llvm_code = llvm_gen.result();
                    println!("LLVM code:\n{}\n", llvm_code);

                    if let Err(e) = load_runtime_library() {
                        println!("Couldn't load runtime: {}", e);
                        continue;
                    }

                    if let Err(ref e) = easy_ll::compile_module(&llvm_code, Some(MERGER_BC)) {
                        println!("Error during LLVM compilation:\n{}\n", e);
                    } else {
                        println!("LLVM module compiled successfully\n");
                    }
                }
            }
            Err(ref e) => {
                println!("Error during SIR code gen:\n{}\n", e);
            }
        }
    }
    rl.save_history(&history_file_path).unwrap();
}
