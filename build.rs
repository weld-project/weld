use std::env;
use std::process::Command;

fn main() {
    let project_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let status = Command::new("make")
        .arg("clean")
        .arg("-C")
        .arg(format!("{}/weld_rt/cpp/", project_dir))
        .status()
        .unwrap();
    assert!(status.success());

    let status = Command::new("make")
        .arg("-C")
        .arg(format!("{}/weld_rt/cpp/", project_dir))
        .status()
        .unwrap();
    assert!(status.success());

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

    // Link the weldrt C++ library
    println!("cargo:rustc-link-lib=static=weldrt");
    println!("cargo:rustc-link-search=native={}/weld_rt/cpp", project_dir);

    // Link the dictst C++ library
    println!("cargo:rustc-link-lib=static=dictst");
    println!("cargo:rustc-link-search=native={}/weld_rt/cpp/st", project_dir);

    // Link the LLVM extensions library
    println!("cargo:rustc-link-lib=static=llvmext");
    println!("cargo:rustc-link-search=native={}/llvmext", project_dir);
}
