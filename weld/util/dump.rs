//! Utilities for dumping code to a file.
//!
//! This module can be used both for writing code for consumption in another program (e.g., writing
//! LLVM files that can then be passed to Clang) or for debugging.

use error::*;

// For dumping files.
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Write;
use std::io::Error as IOError;
use std::path::PathBuf;

impl From<IOError> for WeldCompileError {
    fn from(err: IOError) -> WeldCompileError {
        WeldCompileError::new(err.to_string())
    }
}

#[derive(Copy,Clone,Debug,Hash,PartialEq,Eq)]
pub enum DumpCodeFormat {
    Weld,
    WeldOpt,
    LLVM,
    LLVMOpt,
    SIR,
    Assembly,
}

impl DumpCodeFormat {

    /// Returns a vector with all formats.
    pub fn all() -> Vec<DumpCodeFormat> {
        use self::DumpCodeFormat::*;
        vec![Weld, WeldOpt, LLVM, LLVMOpt, SIR, Assembly]
    }

    /// Returns a filename suffix for the format.
    pub fn suffix(&self) -> String {
        use self::DumpCodeFormat::*;
        match self {
            WeldOpt | LLVMOpt => "-opt",
            _ => ""
        }.to_string()
    }

    /// Returns a filename extension for the format.
    pub fn extension(&self) -> String {
        use self::DumpCodeFormat::*;
        match self {
            Weld | WeldOpt => "weld",
            LLVM | LLVMOpt => "ll",
            SIR => "sir",
            Assembly => "S",
        }.to_string()
    }
}

/// Configuration for dumping code.
#[derive(Clone,Debug)]
pub struct DumpCodeConfig {
    pub enabled: bool,
    pub filename: String,
    pub directory: String,
    pub formats: HashSet<DumpCodeFormat>,
}

/// Writes code to a file using the given configuration.
///
/// The format determines the extension used for the dumped file.
pub fn write_code<T: AsRef<str>>(code: T,
                                 format: DumpCodeFormat,
                                 config: &DumpCodeConfig) -> WeldResult<()> {

    // Code dumping is not enabled - return.
    if !config.enabled {
        return Ok(())
    }

    // Format not registered: return.
    if !config.formats.contains(&format) {
        return Ok(())
    }

    let ref mut path = PathBuf::new();
    path.push(&config.directory);
    path.push(&format!("{}{}", &config.filename, format.suffix()));
    path.set_extension(format.extension());

    info!("Writing code to {}", path.to_str().unwrap());

    let mut options = OpenOptions::new();
    let mut file = options.write(true)
        .create_new(true)
        .open(path)?;

    file.write_all(code.as_ref().as_bytes())?; 
    Ok(())
}
