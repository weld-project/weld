use std::process::Command;
use std::env;

fn main() {
    let mut path = match env::var("WELD_HOME") {
        Ok(val) => val,
        Err(_) => "..".to_string(),
    };

    if path.chars().last().unwrap() != '/' {
        path = path + &"/";
    }

    let cpp_path = path + &"weld_rt/cpp";

    Command::new("make").arg("clean").arg("-C").arg(cpp_path.clone()).status().unwrap();
    Command::new("make").arg("-C").arg(cpp_path.clone()).status().unwrap();
    println!("{}", format!("cargo:rustc-link-search=native={}", cpp_path));

    let target = env::var("TARGET").unwrap();
    if target == "x86_64-apple-darwin" {
        let libs = vec!["z", "c++"];
        for lib in libs {
            println!("cargo:rustc-link-lib={}", lib);
        }
    }
}
