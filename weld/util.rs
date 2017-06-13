
extern crate easy_ll;
extern crate libc;

use std;

use std::cmp::max;
use std::collections::HashMap;

use std::env;

use super::ast::*;
use super::ast::ExprKind::*;

pub const MERGER_BC: &'static [u8] = include_bytes!("../weld_rt/cpp/libparbuilder.bc");
const WELD_HOME: &'static str = "WELD_HOME";

/// Utility struct that can track and generate unique IDs and symbols for use in an expression.
/// Each SymbolGenerator tracks the maximum ID used for every symbol name, and can be used to
/// create new symbols with the same name but a unique ID.
pub struct SymbolGenerator {
    id_map: HashMap<String, i32>,
}

impl SymbolGenerator {
    /// Initialize a SymbolGenerator with no existing symbols.
    pub fn new() -> SymbolGenerator {
        SymbolGenerator { id_map: HashMap::new() }
    }

    /// Initialize a SymbolGenerator from all the symbols defined in an expression.
    pub fn from_expression<T: TypeBounds>(expr: &Expr<T>) -> SymbolGenerator {
        let mut id_map: HashMap<String, i32> = HashMap::new();

        let update_id = |id_map: &mut HashMap<String, i32>, symbol: &Symbol| {
            let id = id_map.entry(symbol.name.clone()).or_insert(0);
            *id = max(*id, symbol.id);
        };

        expr.traverse(&mut |e| match e.kind {
                               Let { ref name, .. } => update_id(&mut id_map, name),
                               Ident(ref sym) => update_id(&mut id_map, sym),
                               Lambda { ref params, .. } => {
                                   for ref p in params {
                                       update_id(&mut id_map, &p.name);
                                   }
                               }
                               _ => {}
                           });

        SymbolGenerator { id_map: id_map }
    }

    pub fn new_symbol(&mut self, name: &str) -> Symbol {
        let id = self.id_map.entry(name.to_owned()).or_insert(-1);
        *id += 1;
        Symbol {
            name: name.to_owned(),
            id: *id,
        }
    }

    /// Return the next ID that will be given to a symbol with the given string name.
    pub fn next_id(&self, name: &str) -> i32 {
        match self.id_map.get(name) {
            Some(id) => id + 1,
            None => 0,
        }
    }
}

/// Utility struct to generate string IDs with a given prefix.
pub struct IdGenerator {
    prefix: String,
    next_id: i32,
}

impl IdGenerator {
    /// Initialize an IdGenerator that will begin counting up from 0.
    pub fn new(prefix: &str) -> IdGenerator {
        IdGenerator {
            prefix: String::from(prefix),
            next_id: 0,
        }
    }

    /// Generate a new ID.
    pub fn next(&mut self) -> String {
        let res = format!("{}{}", self.prefix, self.next_id);
        self.next_id += 1;
        res
    }
}

/// Returns the value of the WELD_HOME environment variable,
/// or an error if the variable is not set.
///
/// The returned path has a trailing `/`.
pub fn get_weld_home() -> Result<String, ()> {
    match env::var(WELD_HOME) {
        Ok(path) => {
            let path = if path.chars().last().unwrap() != '/' {
                path + &"/"
            } else {
                path
            };
            Ok(path)
        }
        Err(_) => Err(()),
    }
}

/// Loads the Weld runtime library.
pub fn load_runtime_library() -> Result<(), String> {
    let weld_home = get_weld_home().unwrap_or("./".to_string());
    let path = format!("{}{}", weld_home, "weld_rt/target/release/libweldrt");
    if let Err(_) = easy_ll::load_library(&path) {
        let err_message = unsafe { std::ffi::CStr::from_ptr(libc::dlerror()) };
        let err_message = err_message.to_owned().into_string().unwrap();
        Err(err_message)
    } else {
        Ok(())
    }
}
