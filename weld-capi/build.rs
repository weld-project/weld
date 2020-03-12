use cbindgen;
use std::env;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let header = r##"
#ifdef __cplusplus
extern "C" {
#endif
"##
    .trim();

    let trailer = r##"
#ifdef __cplusplus
}
#endif
"##
    .trim();

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_language(cbindgen::Language::C)
        .with_include_guard("_WELD_H_")
        .with_header(header)
        .with_trailer(trailer)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("weld.h");
}
