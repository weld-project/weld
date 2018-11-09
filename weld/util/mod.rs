//! Compiler utilities.
//!
//! This module contains a number of utilities used throughput the compiler, such as unique symbol
//! and ID generators and a module for measuring compile-time performance of various aspects of the
//! compiler.

extern crate libc;
extern crate fnv;
extern crate time;
extern crate uuid;

use ast::*;
use ast::ExprKind::*;

use std::iter;
use std::cmp::max;

// For dumping files.
use std::io::Write;
use std::path::PathBuf;
use std::fs::OpenOptions;
use uuid::Uuid;

pub mod stats;
pub mod colors;

/// Utility struct that can track and generate unique IDs and symbols for use in an expression.
/// Each SymbolGenerator tracks the maximum ID used for every symbol name, and can be used to
/// create new symbols with the same name but a unique ID.
#[derive(Debug,Clone)]
pub struct SymbolGenerator {
    id_map: fnv::FnvHashMap<String, i32>,
}

impl SymbolGenerator {
    /// Initialize a SymbolGenerator with no existing symbols.
    pub fn new() -> SymbolGenerator {
        SymbolGenerator { id_map: fnv::FnvHashMap::default() }
    }

    /// Initialize a SymbolGenerator from all the symbols defined in an expression.
    pub fn from_expression(expr: &Expr) -> SymbolGenerator {
        let mut id_map: fnv::FnvHashMap<String, i32> = fnv::FnvHashMap::default();

        let update_id = |id_map: &mut fnv::FnvHashMap<String, i32>, symbol: &Symbol| {
            let id = id_map.entry(symbol.name().clone()).or_insert(0);
            *id = max(*id, symbol.id());
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
        Symbol::new(name, *id)
    }
}

pub fn join<T: iter::Iterator<Item = String>>(start: &str, sep: &str, end: &str, strings: T) -> String {
    let mut res = String::new();
    res.push_str(start);
    for (i, s) in strings.enumerate() {
        if i > 0 {
            res.push_str(sep);
        }
        res.push_str(&s);
    }
    res.push_str(end);
    res
}

/// Return a timestamp-based filename for `dumpCode`.
///
/// The timestamp has a 2-character identifier attached to prevent naming conflicts.
pub fn timestamp_unique() -> String {
    let uuid = Uuid::new_v4().to_simple().to_string();
    let ref suffix = uuid[0..2];
    format!("{}-{}", time::now().to_timespec().sec, suffix)
}


/// Writes code to a file specified by `PathBuf`. Writes a log message if it failed.
pub fn write_code<T: AsRef<str>, U: AsRef<str>, V: AsRef<str>>(code: T,
                                                               ext: U,
                                                               prefix: V,
                                                               dir_path: &str) {

    let ref mut path = PathBuf::new();
    path.push(dir_path);

    let mut options = OpenOptions::new();
    options.write(true)
        .create_new(true);

    path.push(format!("code-{}", prefix.as_ref()));
    path.set_extension(ext.as_ref());

    let ref path_str = format!("{}", path.display());
    match options.open(path) {
        Ok(ref mut file) => {
            if let Err(_) = file.write_all(code.as_ref().as_bytes()) {
                error!("Write failed: could not write code to file {}", path_str);
            }
        }
        Err(_) => {
            error!("Open failed: could not write code to file {}", path_str);
        }
    }
}
