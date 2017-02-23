// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp;
use std::mem::forget;
use std::ptr;
use std::time::{Instant, Duration};

pub mod stats;

/// Manager of the benchmarking runs.
///
/// This is fed into functions marked with `#[bench]` to allow for
/// set-up & tear-down before running a piece of code repeatedly via a
/// call to `iter`.
#[derive(Clone)]
pub struct Bencher {
    iterations: u64,
    samples: Vec<f64>,
    dur: Duration,
    pub bytes: u64,
}

#[derive(Clone, PartialEq)]
pub struct BenchSamples {
    pub ns_iter_summ: stats::Summary,
    pub mb_s: usize,
}

// Benchmarking

// FIXME: We don't have black_box in stable rust

/// NOTE: We don't have a proper black box in stable Rust. This is
/// a workaround implementation, that may have a too big performance overhead,
/// depending on operation, or it may fail to properly avoid having code
/// optimized out. It is good enough that it is used by default.
///
/// A function that is opaque to the optimizer, to allow benchmarks to
/// pretend to use outputs to assist in avoiding dead-code
/// elimination.
pub fn black_box<T>(dummy: T) -> T {
    unsafe {
        let ret = ptr::read_volatile(&dummy);
        forget(dummy);
        ret
    }
}

impl Bencher {
    /// Returns a new `Bencher`.
    pub fn new() -> Bencher {
        Bencher {
            iterations: 10,
            samples: vec![],
            dur: Duration::new(0, 0),
            bytes: 0,
        }
    }

    /// Callback for benchmark functions to run in their body.
    /// This is the part that is timed.
    pub fn iter<T, F>(&mut self, mut inner: F)
        where F: FnMut() -> T
    {
        for _ in 0..self.iterations {
            let start = Instant::now();
            black_box(inner());
            self.dur = start.elapsed();
            let sample = self.ns_elapsed();
            self.samples.push(sample);
        }
    }

    /// Returns the ns elapsed for the last benchmark run.
    pub fn ns_elapsed(&self) -> f64 {
        (self.dur.as_secs() * 1_000_000_000 + (self.dur.subsec_nanos() as u64)) as f64
    }

    pub fn bench_run<F>(&mut self, f: F)
        where F: FnOnce(&mut Bencher)
    {
        f(self);
    }

    pub fn auto_bench<F>(&mut self, mut f: F) -> stats::Summary
        where F: FnMut(&mut Bencher)
    {
        // Initial bench run to get ballpark figure.
        self.bench_run(|x| f(x));

        // If one run took more than 500 ms, just stop.
        if self.ns_elapsed() > (500_000_000 as f64) {
            // stats::winsorize(&mut self.samples, 5.0);
            let summ = stats::Summary::new(self.samples.as_slice());
            return summ;
        }

        // Otherwise, run to get more samples.
        self.bench_run(|x| f(x));

        // stats::winsorize(&mut self.samples, 5.0);
        let summ = stats::Summary::new(self.samples.as_slice());
        return summ;
    }
}

pub mod bench {
    use std::cmp;
    use super::{Bencher, BenchSamples};

    pub fn benchmark<F>(f: F) -> BenchSamples
        where F: FnMut(&mut Bencher)
    {
        let mut bs = Bencher::new();

        let ns_iter_summ = bs.auto_bench(f);

        let ns_iter = cmp::max(ns_iter_summ.median as u64, 1);
        let mb_s = bs.bytes * 1000 / ns_iter;

        BenchSamples {
            ns_iter_summ: ns_iter_summ,
            mb_s: mb_s as usize,
        }
    }
}
