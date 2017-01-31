

A port of the libtest (unstable Rust) benchmark runner to Rust stable releases.
Supports running benchmarks and filtering based on the name. Benchmark
execution works exactly the same way and no more (Warning: black_box is not
working perfectly!).

Please read the `API documentation here`__ (it includes a usage example).

__ https://docs.rs/bencher/

|build_status|_ |crates|_

.. |build_status| image:: https://travis-ci.org/bluss/bencher.svg?branch=master
.. _build_status: https://travis-ci.org/bluss/bencher

.. |crates| image:: https://meritbadge.herokuapp.com/bencher
.. _crates: https://crates.io/crates/bencher

Recent Changes
--------------

- 0.1.2

  - Remove unused components (speeds up build time of the crate)

- 0.1.1

  - Add a provisional implementation of ``black_box``. It's not as good as the
    original version. (Since reproducibility is key, we will use the same
    implementation on both stable and nightly.)
  - Add example for how to set up this to run with ``cargo bench`` on stable.
    This crate is itself an example of that, see ``Cargo.toml`` and ``benches/``

- 0.1.0

  - Initial release

Authors
-------

Principal original authors of the benchmark and statistics code in the Rust
project are:

+ Brian Anderson
+ Graydon Hoare

Very very many have contributed to lib.rs and stats.rs however, so author
credit is due to:

+ The Rust Project Developers

License
-------

Dual-licensed just like the Rust project.

Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0 or the MIT license
http://opensource.org/licenses/MIT, at your
option. This file may not be copied, modified, or distributed
except according to those terms.
