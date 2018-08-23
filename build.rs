extern crate uuid;

use std::env;
use std::process::Command;
use uuid::Uuid;

use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    let project_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    // For writing a UUID for this build.
    let out_dir = env::var("OUT_DIR").unwrap();
    // Create a new MD5 hash for this build.

    let dest_path = Path::new(&out_dir).join("build_uuid");
    let mut f = File::create(&dest_path).unwrap();

    let uuid = Uuid::new_v4();
    let as_string = format!("{}", uuid);
    f.write_all(as_string.as_bytes()).unwrap();

    let status = Command::new("make")
        .arg("-C")
        .arg(format!("{}/weld_rt/cpp/st", project_dir))
        .status()
        .unwrap();
    assert!(status.success());

    let status = Command::new("make")
        .arg("-C")
        .arg(format!("{}/llvmext/", project_dir))
        .status()
        .unwrap();
    assert!(status.success());

    // Link C++ standard library and some Mac-specific libraries
    let target = env::var("TARGET").unwrap();
    if target == "x86_64-apple-darwin" {
        let libs = vec!["z", "c++"];
        for lib in libs {
            println!("cargo:rustc-link-lib={}", lib);
        }
    }
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // Link the dictst C++ library
    println!("cargo:rustc-link-lib=static=dictst");
    println!("cargo:rustc-link-search=native={}/weld_rt/cpp/st", project_dir);

    // Link the LLVM extensions library
    println!("cargo:rustc-link-lib=static=llvmext");
    println!("cargo:rustc-link-search=native={}/llvmext", project_dir);
}
