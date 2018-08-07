//! Compiler utilities.
//!
//! This module contains a number of utilities used throughput the compiler, such as unique symbol
//! and ID generators and a module for measuring compile-time performance of various aspects of the
//! compiler.

extern crate libc;
extern crate fnv;

use ast::*;
use ast::ExprKind::*;

use std::iter;
use std::cmp::max;

// For dumping files.
use std::io::Write;
use std::path::PathBuf;
use std::fs::OpenOptions;

pub mod stats;
pub mod colors;

/// Utility struct that can track and generate unique IDs and symbols for use in an expression.
/// Each SymbolGenerator tracks the maximum ID used for every symbol name, and can be used to
/// create new symbols with the same name but a unique ID.
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


/// Writes code to a file specified by `PathBuf`. Writes a log message if it failed.
pub fn write_code<T: AsRef<str>, U: AsRef<str>, V: AsRef<str>>(code: T, ext: U, prefix: V, dir_path: &PathBuf) {
    let mut options = OpenOptions::new();
    options.write(true)
        .create_new(true)
        .create(true);

    let ref mut path = dir_path.clone();
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
