use std::process::Command;

fn main() {
    Command::new("make").arg("clean").status().unwrap();
    Command::new("make").arg("-C").arg("cpp").status().unwrap();
    println!("cargo:rustc-link-search=native=cpp");
}
