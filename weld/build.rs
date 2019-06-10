use std::env;
use std::process::Command;

/// Write the build ID into an environment variable.
fn register_build_id() {
    // Set the build ID, which is either unknown or the git hash
    let unknown_build = String::from("unknown");
    let build_id = match Command::new("git")
        .arg("rev-parse")
        .arg("--short")
        .arg("HEAD")
        .output()
    {
        Ok(output) => String::from_utf8(output.stdout).unwrap_or(unknown_build),
        Err(_) => unknown_build,
    };
    println!("cargo:rustc-env=BUILD_ID={}", build_id);
}

/// Link the C++ libraries.
fn link_stdcpp() {
    // Link C++ standard library and some Mac-specific libraries
    let target = env::var("TARGET").unwrap();
    if target == "x86_64-apple-darwin" {
        let libs = vec!["z", "c++"];
        for lib in libs {
            println!("cargo:rustc-link-lib={}", lib);
        }
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

/// Build the LLVM Extensions.
fn build_llvmext(project_dir: &str) {
    let status = Command::new("make")
        .arg("-C")
        .arg(format!("{}/llvmext/", project_dir))
        .status()
        .unwrap();
    assert!(status.success());

    let ref out_dir = env::var("OUT_DIR").unwrap();
    println!("cargo:rustc-link-lib=static=llvmext");
    println!("cargo:rustc-link-search=native={}", out_dir);
}

fn main() {
    let ref project_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    // Build ID
    register_build_id();

    // Link stdlibc++: We need this for LLVM.
    link_stdcpp();

    // Build and link external libs.
    build_llvmext(project_dir);
}
