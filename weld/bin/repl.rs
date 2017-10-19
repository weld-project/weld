extern crate rustyline;
extern crate weld;
extern crate libc;

#[macro_use]
extern crate lazy_static;

extern crate clap;
use clap::{Arg, App};

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

const PROMPT: &'static str = ">>> ";

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

lazy_static! {
    static ref RESERVED_WORDS: HashMap<String, ReplCommands> = {
        let mut m = HashMap::new();
        m.insert(ReplCommands::LoadFile.to_string(), ReplCommands::LoadFile);
        m.insert(ReplCommands::SetConf.to_string(), ReplCommands::SetConf);
        m.insert(ReplCommands::GetConf.to_string(), ReplCommands::GetConf);
        m
    };
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

/// Reads a line of input, returning the read line or `None` if an error occurred
/// or the user exits.
fn read_input(rl: &mut Editor<()>, prompt: &str, history: bool) -> Option<String> {
    let raw_readline = rl.readline(prompt);
    match raw_readline {
        Ok(raw_readline) => {
            if history {
                rl.add_history_entry(&raw_readline);
            }
            let trimmed = raw_readline.trim();
            Some(trimmed.to_string())
        }
        Err(ReadlineError::Interrupted) => {
            println!("Exiting!");
            None
        }
        Err(ReadlineError::Eof) => {
            println!("Exiting!");
            None
        }
        Err(err) => {
            println!("Error: {:?}", err);
            None
        }
    }
}

/// Handles a single string command. Returns a string if the command
/// contains code or `None` if the command is fully processed.
fn handle_string<'a>(command: &'a str, conf: *mut WeldConf) -> Option<String> {
    let mut tokens = command.splitn(2, " ");
    let command = tokens.next().unwrap();
    let arg = tokens.next().unwrap_or("");
    if RESERVED_WORDS.contains_key(command) {
        let command = RESERVED_WORDS.get(command).unwrap();
        match *command {
            ReplCommands::LoadFile => {
                match process_loadfile(arg.to_string()) {
                    Err(s) => {
                        println!("{}", s);
                        None
                    }
                    Ok(code) => {
                        Some(code)
                    }
                }
            }
            ReplCommands::SetConf => {
                let mut setconf_args = arg.splitn(2, " ");
                let key = setconf_args.next().unwrap_or("");
                let value = setconf_args.next().unwrap_or("");
                process_setconf(conf, key.to_string(), value.to_string());
                None
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
                None
            }
        }
    } else {
        Some(command.to_string())
    }
}

fn main() {
    let matches = App::new("Weld REPL")
        .version("0.1.0")
        .author("Weld authors <weld-group@cs.stanford.edu")
        .about("A REPL for Weld")
        .arg(Arg::with_name("loglevel")
             .short("l")
             .long("loglevel")
             .value_name("LEVEL")
             .help("Log level for the Weld compiler")
             .takes_value(true))
        .get_matches();

    let log_level_str = matches.value_of("loglevel").unwrap_or("debug").to_lowercase();
    let (log_level, log_str) = match log_level_str.as_str() {
        "none" =>   (WeldLogLevel::Off,         "none"),
        "error" =>  (WeldLogLevel::Error,       "\x1b[0;31merror\x1b[0m"),
        "warn" =>   (WeldLogLevel::Warn,        "\x1b[0;33mwarn\x1b[0m"),
        "info" =>   (WeldLogLevel::Info,        "\x1b[0;33minfo\x1b[0m"),
        "debug" =>  (WeldLogLevel::Debug,       "\x1b[0;32mdebug\x1b[0m"), 
        "trace" =>  (WeldLogLevel::Trace,       "\x1b[0;32mtrace\x1b[0m"),
        ref s => {
            println!("Unrecognized log level {}", s);
            std::process::exit(1);
        }
    };
    weld_set_log_level(log_level);
    println!("Log Level set to '{}'", log_str);

    // This is the conf we use for compilation.
    let conf = weld_conf_new();

    let home_path = env::home_dir().unwrap_or(PathBuf::new());
    let history_file_path = home_path.join(".weld_history");
    let history_file_path = history_file_path.to_str().unwrap_or(".weld_history");

    let mut rl = Editor::<()>::new();
    if let Err(_) = rl.load_history(&history_file_path) {}

    loop {
        // Check if the input was valid.
        let input = read_input(&mut rl, PROMPT, true);
        if input.is_none() {
            break;
        }
        let input = input.unwrap();
        if input == "" {
            continue;
        }

        // Handle repl commands.
        let code = handle_string(&input, conf);
        if code.is_none() {
            continue;
        }
        let code = code.unwrap();

        // Process the code.
        unsafe {
            let code = CString::new(code).unwrap();
            let err = weld_error_new();
            let module = weld_module_compile(code.into_raw() as *const c_char, conf, err);
            if weld_error_code(err) != WeldRuntimeErrno::Success {
                println!("REPL: Compile error: {}",
                    CStr::from_ptr(weld_error_message(err)).to_str().unwrap());
            } else {
                println!("REPL: Program compiled successfully to LLVM");
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
