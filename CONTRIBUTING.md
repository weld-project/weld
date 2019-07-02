When you contribute code, you affirm that the contribution is your original work and that you license the work to the project under the project's open source license. Whether or not you state this explicitly, by submitting any copyrighted material via pull request, email, or other means you agree to license the material under the project's open source license and warrant that you have the legal authority to do so.

## Tests

Weld has several integration and doctests, controlled via `cargo`. Travis CI
ensures that new patches do not fail existing tests. To run these tests locally:

```
cargo test
```

## Formatting

Weld uses `clippy` and `rustfmt` for lints and format checks respectively.
Travis CI checks to ensure that all lints pass and that code is formatted in
accordance with `rustfmt`. To check this before submitting code, run the
following:

```
cargo clippy
cargo fmt -- --check
```
