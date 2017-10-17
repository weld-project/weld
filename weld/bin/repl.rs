extern crate rustyline;
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
use std::ffi::{CStr, CString};
use std::collections::HashMap;
use libc::c_char;

use weld::*;
use weld::common::*;

enum ReplCommands {
    LoadFile,
    GetConf,
    SetConf,
}

impl fmt::Display for ReplCommands {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ReplCommands::LoadFile => write!(f, "load"),
            ReplCommands::GetConf => write!(f, "getconf"),
            ReplCommands::SetConf => write!(f, "setconf"),
        }
    }
}

/// Process the `SetConf` command.
///
/// The argument is a key/value pair. The command sets the key/value pair for the REPL's
/// configuration.
fn process_setconf(conf: *mut WeldConf, key: String, value: String) {
    let key = CString::new(key).unwrap();
    let value = CString::new(value).unwrap();
    unsafe {
        weld_conf_set(conf, key.as_ptr(), value.as_ptr());
    }
}

/// Process the `GetConf` command.
///
/// The argument is a key in the configuration. The command returns the value of the key or `None`
/// if no value is set.
fn process_getconf(conf: *mut WeldConf, key: String) -> Option<String> {
    let key = CString::new(key).unwrap();
    unsafe {
        let val = weld_conf_get(conf, key.as_ptr());
        if val.is_null() {
            None
        } else {
            let val = CStr::from_ptr(val);
            let val = val.to_str();
            if let Ok(s) = val {
                Some(s.to_string())
            } else {
                None
            }
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

    // This is the conf we use for compilation.
    let conf = weld_conf_new();

    let home_path = env::home_dir().unwrap_or(PathBuf::new());
    let history_file_path = home_path.join(".weld_history");
    let history_file_path = history_file_path.to_str().unwrap_or(".weld_history");

    let mut reserved_words = HashMap::new();
    reserved_words.insert(ReplCommands::LoadFile.to_string(), ReplCommands::LoadFile);
    reserved_words.insert(ReplCommands::SetConf.to_string(), ReplCommands::SetConf);
    reserved_words.insert(ReplCommands::GetConf.to_string(), ReplCommands::GetConf);

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
                ReplCommands::SetConf => {
                    let mut setconf_args = arg.splitn(2, " ");
                    let key = setconf_args.next().unwrap_or("");
                    let value = setconf_args.next().unwrap_or("");
                    process_setconf(conf, key.to_string(), value.to_string());
                    "".to_string()
                }
                ReplCommands::GetConf => {
                    let mut setconf_args = arg.splitn(2, " ");
                    let key = setconf_args.next().unwrap_or("");
                    let value = process_getconf(conf, key.to_string());
                    if let Some(s) = value {
                        println!("{}={}", key, s);
                    } else {
                        println!("{}=<unset>", key);
                    }
                    "".to_string()
                }
            }
        } else {
            trimmed.to_string()
        };

        if code.len() == 0 {
            continue;
        }

        unsafe {
            let code = CString::new(code).unwrap();
            let err = weld_error_new();
            let module = weld_module_compile(code.into_raw() as *const c_char, conf, err);
            if weld_error_code(err) != WeldRuntimeErrno::Success {
                println!("Compile error: {}",
                    CStr::from_ptr(weld_error_message(err)).to_str().unwrap());
            } else {
                println!("Program compiled successfully to LLVM");
                weld_module_free(module);
            }
            weld_error_free(err);
        }
    }

    unsafe {
        weld_conf_free(conf);
    }

    rl.save_history(&history_file_path).unwrap();
}
