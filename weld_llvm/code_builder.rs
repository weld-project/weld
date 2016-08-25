//! Utility for generating code that handles some formatting.

// TODO(shoumik): Remove - throwing warnings right now.
#![allow(dead_code)]

use std::iter;
use std::cmp::max;

#[derive(Debug)]
pub struct CodeBuilder {
    code: String,
    indent_level: i32,
    indent_size: i32,
    indent_string: String,
}

/// Methods for `CodeBuilder`.
impl CodeBuilder {
    /// Adds line to code and performs some simple formatting.
    pub fn add_line(&mut self, line: &str) {
        let indent_change = (line.matches("{").count() as i32) -
            (line.matches("}").count() as i32);
        let new_indent_level = max(0, self.indent_level + indent_change);
        // Lines starting with '}' should be de-indented even if they
        // contain '{' after; in addition, lines ending with ':' are
        // typically labels, e.g., in LLVM.
        let this_line_indent = if line.starts_with("}") || line.ends_with(":") {
            let spaces = (self.indent_size * (self.indent_level - 1)) as usize;
            iter::repeat(" ")
                .take(spaces)
                .collect::<String>()
        } else {
            // TODO(shoumik): How to avoid this clone...?
            self.indent_string.clone()
        };

        self.code.push_str(this_line_indent.as_ref());
        self.code.push_str(line.trim());
        self.code.push_str("\n");

        self.indent_level = new_indent_level;
        let spaces = (self.indent_size * new_indent_level) as usize;
        self.indent_string = iter::repeat(" ")
            .take(spaces)
            .collect::<String>();
    }

    /// Adds multiples lines (split by \n) to this code builder.
    pub fn add_lines(&mut self, code: &str) {
        for l in code.lines() {
            self.add_line(l);
        }
    }

    /// Returns the code in this code builder so far.
    pub fn result(&self) -> String {
        self.code.clone()
    }

    /// Returns a new CodeBuilder.
    pub fn new(indent_size: i32) -> CodeBuilder {
        CodeBuilder {
            code: String::new(),
            indent_level: 0,
            indent_size: indent_size,
            indent_string: String::new(),
        }
    }

    /// Returns a formatted string using the CodeBuilder.
    pub fn format(indent_size: i32, code: &str) -> String {
        let mut c = CodeBuilder::new(indent_size);
        c.add_lines(code);
        c.result()
    }
}

#[test]
fn code_builder_basic() {
    let inp = "
class A {
blahblah;
}";
   let exp = "
class A {
  blahblah;
}
";
   assert_eq!(CodeBuilder::format(2, inp), exp);

   let inp = "
class A {
if (c) {
duh
}
}";
   let exp = "
class A {
  if (c) {
    duh
  }
}
";
   assert_eq!(CodeBuilder::format(2, inp), exp);

   let inp = "
class A {
if (c) { duh }
}";
   let exp = "
class A {
  if (c) { duh }
}
";
   assert_eq!(CodeBuilder::format(2, inp), exp);

   let inp = "
class A {
if (c) { duh }
myLabel:
blah
}";
   let exp = "
class A {
  if (c) { duh }
myLabel:
  blah
}
";
   assert_eq!(CodeBuilder::format(2, inp), exp);
}
