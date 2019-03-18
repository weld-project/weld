//! Basic format strings for colored output.
#![allow(dead_code)]

const RESET: &str = "\x1b[0m";

pub enum Color {
    Red,
    BoldRed,
    Green,
    Yellow,
}

trait Prefix {
    fn prefix(&self) -> &str;
}

impl Prefix for Color {
    fn prefix(&self) -> &str {
        match *self {
            Color::Red => "\x1b[0;31m",
            Color::BoldRed => "\x1b[1;31m",
            Color::Green => "\x1b[0;32m",
            Color::Yellow => "\x1b[0;33m",
        }
    }
}

pub fn format_color(color: Color, text: &str) -> String {
    format!("{}{}{}", color.prefix(), text, RESET)
}
