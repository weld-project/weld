extern crate rustyline;
extern crate easy_ll;
extern crate weld;
extern crate weld_common;
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
use std::ffi::{CStr, CString};
use std::collections::HashMap;
use libc::c_char;

use weld::*;
use weld_common::*;

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
    weld_set_log_level(WeldLogLevel::Debug);

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

        // Check whether the command is to load a file; if not, treat it as a program to run.
        let mut tokens = trimmed.splitn(2, " ");
        let command = tokens.next().unwrap();
        let arg = tokens.next().unwrap_or("");
        let code = if reserved_words.contains_key(command) {
            let command = reserved_words.get(command).unwrap();
            match *command {
                ReplCommands::LoadFile => {
                    match process_loadfile(arg.to_string()) {
                        Err(s) => {
                            println!("{}", s);
                            continue;
                        }
                        Ok(code) => {
                            code
                        }
                    }
                }
            }
        } else {
            trimmed.to_string()
        };

        unsafe {
            let code = CString::new(code).unwrap();
            let conf = weld_conf_new();
            let err = weld_error_new();
            let module = weld_module_compile(code.into_raw() as *const c_char, conf, err);
            if weld_error_code(err) != WeldRuntimeErrno::Success {
                println!("Compile error: {}",
                    CStr::from_ptr(weld_error_message(err)).to_str().unwrap());
            } else {
                println!("Program compiled successfully to LLVM");
                weld_module_free(module);
            }
            weld_conf_free(conf);
            weld_error_free(err);
        }
    }
    rl.save_history(&history_file_path).unwrap();
}
