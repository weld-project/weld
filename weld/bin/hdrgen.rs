//! Generates headers for a Weld program.

#[macro_use]
extern crate weld;
extern crate libc;
extern crate clap;

use weld::*;
use weld::common::*;

use clap::{Arg, App};
use libc::c_char;

use std::ffi::{CStr, CString};
use std::collections::HashMap;

use std::path::Path;
use std::fs::File;
use std::error::Error;
use std::io::prelude::*;

use weld::code_builder::CodeBuilder;
use weld::util::IdGenerator;
use weld::ast::*;

static PRELUDE_CODE: &'static str = include_str!("resources/cpp_prelude.h");

#[derive(Debug,Clone,PartialEq,Eq)]
struct LambdaTypes {
    param_types: Vec<Type>,
    return_type: Type,
}

fn parse_weld_lambda(code: &str) -> WeldResult<LambdaTypes> {
    unsafe {
        let code = CString::new(code).unwrap();
        let err = weld_error_new();
        let conf = weld_conf_new();
        let module = weld_module_compile(code.into_raw() as *const c_char, conf, err);
        if weld_error_code(err) != WeldRuntimeErrno::Success {
            return weld_err!("Compile error: {}",
                     CStr::from_ptr(weld_error_message(err)).to_str().unwrap());
        }
        weld_error_free(err);
        weld_conf_free(conf);

        let module_ref = &*module;

        let result = LambdaTypes {
            return_type: module_ref.return_type().clone(),
            param_types: module_ref.param_types().clone(),
        };
        
        // Don't actually need the code, just the types.
        weld_module_free(module);
        Ok(result)
    }
}

struct CppHeaderGenerator {
    /// List of already generated types.
    generated_types: HashMap<Type, String>,
    struct_names: IdGenerator,
    code: CodeBuilder, 
}

impl CppHeaderGenerator {
    /// Return a new header generator for C++.
    fn new() -> CppHeaderGenerator {
        CppHeaderGenerator{
            generated_types: HashMap::new(),
            struct_names: IdGenerator::new("struct"),
            code: CodeBuilder::new(),
        }
    }

    /// Generates a new Struct definition.
    fn generate_struct_definition(&mut self, types: &Vec<Type>) -> WeldResult<String> {
        let struct_name = self.struct_names.next();
        let mut names = vec![];
        for ty in types.iter() {
            names.push(self.generate_type(ty)?);
        }
        self.code.add(format!("struct {} {{", struct_name));
        for (field, name) in names.iter().enumerate() {
            self.code.add(format!("{} _{};", name, field)); 
        }
        self.code.add("};");
        Ok(struct_name)
    }

    /// Generates a single type in returns its name.
    fn generate_type(&mut self, ty: &Type) -> WeldResult<String> {
        use ast::Type::*;
        if let Some(name) = self.generated_types.get(ty) {
            return Ok(name.to_string());
        }

        let result = match *ty {
            Scalar(ref kind) => Ok(format!("{}", kind)),
            Vector(ref elem) => Ok(format!("vec<{}>", self.generate_type(elem)?)),
            Struct(ref elems) => self.generate_struct_definition(elems),
            // Other types (Builders, Functions, etc.) cannot be passed into Weld.
            _ => weld_err!("Invalid C++ type {:?}", ty),
        };

        if result.is_ok() {
            self.generated_types.insert(ty.clone(), result.as_ref().unwrap().clone());
        }
        self.code.add("\n");

        result
    }

    fn build(&mut self, types: LambdaTypes) -> String {
        // Add the prelude code, which defines the templatized vector type vec<T> 
        // and the primitive types (i1, i32, f32, etc.).
        self.code.add("#ifndef _WELD_CPP_HEADER_");
        self.code.add("#define _WELD_CPP_HEADER_");
        self.code.add("\n");
        self.code.add(PRELUDE_CODE);
        self.code.add("\n");

        let return_type = self.generate_type(&types.return_type)
            .expect("Type generation failed!");
        let param_type = self.generate_type(&ast::Type::Struct(types.param_types))
            .expect("Type generation failed!");

        self.code.add("\n");
        self.code.add("// Aliases for argument and return types.");
        self.code.add(format!("typedef {} input_type;", param_type));
        self.code.add(format!("typedef {} return_type;", return_type));
        self.code.add("\n");
        self.code.add("#endif /* _WELD_CPP_HEADER_ */");
        self.code.result().to_string()
    }
}

fn read_full(arg: &str) -> WeldResult<String> {
    let path = Path::new(arg);
    let path_display = path.display();

    let mut file = match File::open(&path) {
        Err(why) => {
            return weld_err!("Error: couldn't open {}: {}",
                             path_display,
                             why.description());
        }
        Ok(res) => res,
    };

    let mut contents = String::new();
    if let Err(why) = file.read_to_string(&mut contents) {
        return weld_err!("Error: couldn't read {}: {}",
                         path_display,
                         why.description());
    }
    Ok(contents.trim().to_string())
}

fn main() {
    let matches = App::new("Weld Header Generator")
        .version("0.1.0")
        .author("Weld authors <weld-group@cs.stanford.edu")
        .about("Generates headers for types which appear in Weld programs")
        .arg(Arg::with_name("input")
             .short("i")
             .long("input")
             .value_name("FILE")
             .help("Weld program to generate a header for")
             .takes_value(true))
        .get_matches();

    let code = read_full(matches.value_of("input").expect("Argument required"))
        .expect("Invalid code file");

    let type_info = parse_weld_lambda(&code)
        .expect("Weld code compilation failed");

    let mut generator = CppHeaderGenerator::new();
    let result = generator.build(type_info);

    println!("{}", result);
}
