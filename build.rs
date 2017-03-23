use std::process::Command;
use std::env;

fn main() {
    let mut path = match env::var("WELD_HOME") {
        Ok(val) => val,
        Err(_) => ".".to_string(),
    };

    if path.chars().last().unwrap() != '/' {
        path = path + &"/";
    }

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

    // NOTE this is pretty hacky...better way?
    let weldrt_path = path + &"weld_rt/Cargo.toml";
    Command::new("cargo")
        .arg("build")
        .arg("--manifest-path")
        .arg(weldrt_path.clone())
        .arg("--release")
        .status()
        .unwrap();
}
