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
#[derive(Copy, Clone)]
pub struct Bencher {
    iterations: u64,
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
    /// Callback for benchmark functions to run in their body.
    pub fn iter<T, F>(&mut self, mut inner: F)
        where F: FnMut() -> T
    {
        let start = Instant::now();
        let k = self.iterations;
        for _ in 0..k {
            black_box(inner());
        }
        self.dur = start.elapsed();
    }

    pub fn ns_elapsed(&mut self) -> u64 {
        self.dur.as_secs() * 1_000_000_000 + (self.dur.subsec_nanos() as u64)
    }

    pub fn ns_per_iter(&mut self) -> u64 {
        if self.iterations == 0 {
            0
        } else {
            self.ns_elapsed() / cmp::max(self.iterations, 1)
        }
    }

    pub fn bench_n<F>(&mut self, n: u64, f: F)
        where F: FnOnce(&mut Bencher)
    {
        self.iterations = n;
        f(self);
    }

    // This is a more statistics-driven benchmark algorithm
    pub fn auto_bench<F>(&mut self, mut f: F) -> stats::Summary
        where F: FnMut(&mut Bencher)
    {
        // Initial bench run to get ballpark figure.
        let mut n = 1;
        self.bench_n(n, |x| f(x));

        // Try to estimate iter count for 1ms falling back to 1m
        // iterations if first run took < 1ns.
        if self.ns_per_iter() == 0 {
            n = 1_000_000;
        } else {
            n = 1_000_000 / cmp::max(self.ns_per_iter(), 1);
        }
        // if the first run took more than 1ms we don't want to just
        // be left doing 0 iterations on every loop. The unfortunate
        // side effect of not being able to do as many runs is
        // automatically handled by the statistical analysis below
        // (i.e. larger error bars).
        if n == 0 {
            n = 1;
        }

        let mut total_run = Duration::new(0, 0);
        let samples: &mut [f64] = &mut [0.0_f64; 50];
        loop {
            let loop_start = Instant::now();

            for p in &mut *samples {
                self.bench_n(n, |x| f(x));
                *p = self.ns_per_iter() as f64;
            }

            stats::winsorize(samples, 5.0);
            let summ = stats::Summary::new(samples);

            for p in &mut *samples {
                self.bench_n(5 * n, |x| f(x));
                *p = self.ns_per_iter() as f64;
            }

            stats::winsorize(samples, 5.0);
            let summ5 = stats::Summary::new(samples);
            let loop_run = loop_start.elapsed();

            // If we've run for 100ms and seem to have converged to a
            // stable median.
            if loop_run > Duration::from_millis(100) && summ.median_abs_dev_pct < 1.0 &&
               summ.median - summ5.median < summ5.median_abs_dev {
                return summ5;
            }

            total_run += loop_run;
            // Longest we ever run for is 3s.
            if total_run > Duration::from_secs(3) {
                return summ5;
            }

            // If we overflow here just return the results so far. We check a
            // multiplier of 10 because we're about to multiply by 2 and the
            // next iteration of the loop will also multiply by 5 (to calculate
            // the summ5 result)
            n = match n.checked_mul(10) {
                Some(_) => n * 2,
                None => return summ5,
            };
        }
    }
}

pub mod bench {
    use std::cmp;
    use std::time::Duration;
    use super::{Bencher, BenchSamples};

    // TODO(shoumik): Header format.
    pub fn benchmark<F>(f: F) -> BenchSamples
        where F: FnMut(&mut Bencher)
    {
        let mut bs = Bencher {
            iterations: 0,
            dur: Duration::new(0, 0),
            bytes: 0,
        };

        let ns_iter_summ = bs.auto_bench(f);

        let ns_iter = cmp::max(ns_iter_summ.median as u64, 1);
        let mb_s = bs.bytes * 1000 / ns_iter;

        BenchSamples {
            ns_iter_summ: ns_iter_summ,
            mb_s: mb_s as usize,
        }

        // TODO(shoumik): Write out the output here.
    }

    #[allow(dead_code)]
    pub fn run_once<F>(f: F)
        where F: FnOnce(&mut Bencher)
    {
        let mut bs = Bencher {
            iterations: 0,
            dur: Duration::new(0, 0),
            bytes: 0,
        };
        bs.bench_n(1, f);
    }
}
