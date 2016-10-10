use std::iter;
use std::cmp::max;

/// Utility struct for generating code that indents and formats it.
#[derive(Debug)]
pub struct CodeBuilder {
    code: String,
    indent_level: i32,
    indent_size: i32,
    indent_string: String,
}

impl CodeBuilder {
    /// Adds a single line of code to this code builder, formatting it based on previous code.
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

    /// Adds one or more lines (split by "\n") to this code builder.
    pub fn add(&mut self, code: &str) {
        for l in code.lines() {
            self.add_line(l);
        }
    }

    /// Adds the result of another CodeBuilder.
    pub fn add_code(&mut self, builder: &CodeBuilder) {
        self.code.push_str(builder.code.as_ref());
    }

    /// Returns the code in this code builder so far.
    pub fn result(&self) -> &str {
        self.code.as_str()
    }

    /// Returns a new CodeBuilder.
    pub fn new() -> CodeBuilder {
        CodeBuilder::with_indent_size(2)
    }

    /// Returns a new CodeBuilder with the given indent size.
    pub fn with_indent_size(indent_size: i32) -> CodeBuilder {
        CodeBuilder {
            code: String::new(),
            indent_level: 0,
            indent_size: indent_size,
            indent_string: String::new(),
        }
    }

    /// Returns a formatted string using the CodeBuilder.
    pub fn format(indent_size: i32, code: &str) -> String {
        let mut c = CodeBuilder::with_indent_size(indent_size);
        c.add(code);
        c.code
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
