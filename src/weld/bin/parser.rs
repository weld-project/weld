extern crate weld;

use std::io::{stdin, stdout, Write};
use weld::grammar::*;

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
            println!("{:?}", parse_expr(trimmed));
        }
    }
}