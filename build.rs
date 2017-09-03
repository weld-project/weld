use std::env;
use std::process::Command;

fn main() {
    Command::new("make")
        .arg("clean")
        .arg("-C")
        .arg("python/grizzly/")
        .status()
        .unwrap();

    Command::new("make")
        .arg("convertor")
        .arg("-C")
        .arg("python/grizzly/")
        .status()
        .unwrap();

    Command::new("make")
        .arg("clean")
        .arg("-C")
        .arg("weld_rt/cpp/")
        .status()
        .unwrap();

    Command::new("make")
        .arg("-C")
        .arg("weld_rt/cpp/")
        .status()
        .unwrap();

    let target = env::var("TARGET").unwrap();
    if target == "x86_64-apple-darwin" {
        let libs = vec!["z", "c++"];
        for lib in libs {
            println!("cargo:rustc-link-lib={}", lib);
        }
    }
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
