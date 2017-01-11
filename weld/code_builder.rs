use std::cmp::max;

/// Utility struct for generating code that indents and formats it.
/// Also implements `std::fmt::Write` to support the `write!` macro.
#[derive(Debug)]
pub struct CodeBuilder {
    code: String,
    indent_level: i32,
    indent_size: i32,
}

impl CodeBuilder {
    /// Adds a single line of code to this code builder, formatting it based on previous code.
    pub fn add_line<S>(&mut self, line: S)
        where S: AsRef<str>
    {
        let line = line.as_ref().trim();
        let indent_change = (line.matches("{").count() as i32) - (line.matches("}").count() as i32);
        let new_indent_level = max(0, self.indent_level + indent_change);

        // Lines starting with '}' should be de-indented even if they contain '{' after; in
        // addition, lines ending with ':' are typically labels, e.g., in LLVM.
        let this_line_indent = if line.starts_with("}") || line.ends_with(":") {
            self.indent_level - 1
        } else {
            self.indent_level
        };

        // self.code.push_str(this_line_indent.as_ref());
        for _ in 0..this_line_indent * self.indent_size {
            self.code.push(' ');
        }
        self.code.push_str(line);
        self.code.push_str("\n");

        self.indent_level = new_indent_level;
    }

    /// Adds one or more lines (split by "\n") to this code builder.
    pub fn add<S>(&mut self, code: S)
        where S: AsRef<str>
    {
        for l in code.as_ref().lines() {
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
        }
    }

    /// Returns a formatted string using the CodeBuilder.
    pub fn format<S>(indent_size: i32, code: S) -> String
        where S: AsRef<str>
    {
        let mut c = CodeBuilder::with_indent_size(indent_size);
        c.add(code);
        c.code
    }
}

#[test]
fn code_builder_basic() {
    let input = "
class A {
blahblah;
}";
    let expected = "
class A {
  blahblah;
}
";
    assert_eq!(CodeBuilder::format(2, input), expected);

    let input = "
class A {
if (c) {
duh
     }
}";
    let expected = "
class A {
  if (c) {
    duh
  }
}
";
    assert_eq!(CodeBuilder::format(2, input), expected);

    let input = "
class A {
if (c) { duh }
}";
    let expected = "
class A {
  if (c) { duh }
}
";
    assert_eq!(CodeBuilder::format(2, input), expected);

    let input = "
class A {
if (c) { duh }
myLabel:
blah
}";
    let expected = "
class A {
  if (c) { duh }
myLabel:
  blah
}
";
    assert_eq!(CodeBuilder::format(2, input), expected);
}
