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

    println!("cargo:rustc-link-lib=dylib=stdc++");
}
