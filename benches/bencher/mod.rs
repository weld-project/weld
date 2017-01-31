// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Simplified stable-compatible benchmark runner.
//!
//! Almost all user code will only be interested in `Bencher` and the
//! macros that are used to describe benchmarker functions and
//! the benchmark runner.
//!
//! NOTE: There's no proper `black_box` yet in this stable port of the
//! benchmark runner, only a workaround implementation. It may not work
//! exactly like the upstream `test::black_box`.
//!
//! One way to use this crate is to use it as dev-dependency and setup
//! cargo to compile a file in `benches/` that runs without the testing harness.
//!
//! In Cargo.toml:
//!
//! ```ignore
//! [[bench]]
//! name = "example"
//! harness = false
//! ```
//!
//! In benches/example.rs:
//!
//! ```
//! #[macro_use]
//! extern crate bencher;
//!
//! use bencher::Bencher;
//!
//! fn a(bench: &mut Bencher) {
//!     bench.iter(|| {
//!         (0..1000).fold(0, |x, y| x + y)
//!     })
//! }
//!
//! fn b(bench: &mut Bencher) {
//!     const N: usize = 1024;
//!     bench.iter(|| {
//!         vec![0u8; N]
//!     });
//!
//!     bench.bytes = N as u64;
//! }
//!
//! benchmark_group!(benches, a, b);
//! benchmark_main!(benches);
//!
//! # #[cfg(never)]
//! # fn main() { }
//! ```
//!
//! Use `cargo bench` as usual. A command line argument can be used to filter
//! which benchmarks to run.

pub use self::TestFn::*;
use self::TestResult::*;
use self::TestEvent::*;
use self::NamePadding::*;
use self::OutputLocation::*;

use std::borrow::Cow;
use std::cmp;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use std::io;
use std::iter::repeat;
use std::mem::forget;
use std::path::PathBuf;
use std::ptr;
use std::time::{Instant, Duration};

mod stats;
mod macros;

// The name of a test. By convention this follows the rules for rust
// paths; i.e. it should be a series of identifiers separated by double
// colons. This way if some test runner wants to arrange the tests
// hierarchically it may.

pub type TestName = Cow<'static, str>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum NamePadding {
    PadOnRight,
}

impl TestDesc {
    fn padded_name(&self, column_count: usize, align: NamePadding) -> String {
        let mut name = self.name.to_string();
        let fill = column_count.saturating_sub(name.len());
        let pad = repeat(" ").take(fill).collect::<String>();
        match align {
            PadOnRight => {
                name.push_str(&pad);
                name
            }
        }
    }
}

/// Represents a benchmark function.
pub trait TDynBenchFn: Send {
    fn run(&self, harness: &mut Bencher);
}

// A function that runs a test. If the function returns successfully,
// the test succeeds; if the function panics then the test fails. We
// may need to come up with a more clever definition of test in order
// to support isolation of tests into threads.
pub enum TestFn {
    StaticBenchFn(fn(&mut Bencher)),
    DynBenchFn(Box<TDynBenchFn + 'static>),
}

impl TestFn {
    fn padding(&self) -> NamePadding {
        match *self {
            StaticBenchFn(..) |
            DynBenchFn(..) => PadOnRight,
        }
    }
}

impl fmt::Debug for TestFn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match *self {
            StaticBenchFn(..) => "StaticBenchFn(..)",
            DynBenchFn(..) => "DynBenchFn(..)",
        })
    }
}

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

// The definition of a single test. A test runner will run a list of
// these.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TestDesc {
    pub name: TestName,
    pub ignore: bool,
}

#[derive(Clone)]
pub struct TestPaths {
    pub file: PathBuf, // e.g., compile-test/foo/bar/baz.rs
    pub base: PathBuf, // e.g., compile-test, auxiliary
    pub relative_dir: PathBuf, // e.g., foo/bar
}

#[derive(Debug)]
pub struct TestDescAndFn {
    pub desc: TestDesc,
    pub testfn: TestFn,
}

#[derive(Default)]
pub struct TestOpts {
    pub filter: Option<String>,
    pub run_ignored: bool,
    pub logfile: Option<PathBuf>,
    pub quiet: bool,
    pub test_threads: Option<usize>,
}

#[derive(Clone, PartialEq)]
pub struct BenchSamples {
    pub ns_iter_summ: stats::Summary,
    pub mb_s: usize,
}

#[derive(Clone, PartialEq)]
enum TestResult {
    TrIgnored,
    TrBench(BenchSamples),
}

unsafe impl Send for TestResult {}

enum OutputLocation<T> {
    Raw(T),
}

struct ConsoleTestState<T> {
    log_out: Option<File>,
    out: OutputLocation<T>,
    quiet: bool,
    total: usize,
    passed: usize,
    failed: usize,
    ignored: usize,
    measured: usize,
    failures: Vec<(TestDesc, Vec<u8>)>,
    max_name_len: usize, // number of columns to fill when aligning names
}

impl ConsoleTestState<()> {
    pub fn new(opts: &TestOpts) -> io::Result<ConsoleTestState<io::Stdout>> {
        let log_out = match opts.logfile {
            Some(ref path) => Some(try!(File::create(path))),
            None => None,
        };
        let out = Raw(io::stdout());

        Ok(ConsoleTestState {
            out: out,
            log_out: log_out,
            quiet: opts.quiet,
            total: 0,
            passed: 0,
            failed: 0,
            ignored: 0,
            measured: 0,
            failures: Vec::new(),
            max_name_len: 0,
        })
    }
}

impl<T: Write> ConsoleTestState<T> {
    pub fn write_ignored(&mut self) -> io::Result<()> {
        self.write_short_result("ignored", "i")
    }

    pub fn write_bench(&mut self) -> io::Result<()> {
        self.write_pretty("bench")
    }

    pub fn write_short_result(&mut self, verbose: &str, quiet: &str) -> io::Result<()> {
        if self.quiet {
            self.write_pretty(quiet)
        } else {
            try!(self.write_pretty(verbose));
            self.write_plain("\n")
        }
    }

    pub fn write_pretty(&mut self, word: &str) -> io::Result<()> {
        match self.out {
            Raw(ref mut stdout) => {
                try!(stdout.write_all(word.as_bytes()));
                stdout.flush()
            }
        }
    }

    pub fn write_plain(&mut self, s: &str) -> io::Result<()> {
        match self.out {
            Raw(ref mut stdout) => {
                try!(stdout.write_all(s.as_bytes()));
                stdout.flush()
            }
        }
    }

    pub fn write_run_start(&mut self, len: usize) -> io::Result<()> {
        self.total = len;
        let noun = if len != 1 { "tests" } else { "test" };
        self.write_plain(&format!("\nrunning {} {}\n", len, noun))
    }

    pub fn write_test_start(&mut self, test: &TestDesc, align: NamePadding) -> io::Result<()> {
        if self.quiet && align != PadOnRight {
            Ok(())
        } else {
            let name = test.padded_name(self.max_name_len, align);
            self.write_plain(&format!("test {} ... ", name))
        }
    }

    pub fn write_result(&mut self, result: &TestResult) -> io::Result<()> {
        match *result {
            TrIgnored => self.write_ignored(),
            TrBench(ref bs) => {
                try!(self.write_bench());
                self.write_plain(&format!(": {}\n", fmt_bench_samples(bs)))
            }
        }
    }

    pub fn write_log(&mut self, test: &TestDesc, result: &TestResult) -> io::Result<()> {
        match self.log_out {
            None => Ok(()),
            Some(ref mut o) => {
                let s = format!("{} {}\n",
                                match *result {
                                    TrIgnored => "ignored".to_owned(),
                                    TrBench(ref bs) => fmt_bench_samples(bs),
                                },
                                test.name);
                o.write_all(s.as_bytes())
            }
        }
    }

    pub fn write_failures(&mut self) -> io::Result<()> {
        try!(self.write_plain("\nfailures:\n"));
        let mut failures = Vec::new();
        let mut fail_out = String::new();
        for &(ref f, ref stdout) in &self.failures {
            failures.push(f.name.to_string());
            if !stdout.is_empty() {
                fail_out.push_str(&format!("---- {} stdout ----\n\t", f.name));
                let output = String::from_utf8_lossy(stdout);
                fail_out.push_str(&output);
                fail_out.push_str("\n");
            }
        }
        if !fail_out.is_empty() {
            try!(self.write_plain("\n"));
            try!(self.write_plain(&fail_out));
        }

        try!(self.write_plain("\nfailures:\n"));
        failures.sort();
        for name in &failures {
            try!(self.write_plain(&format!("    {}\n", name)));
        }
        Ok(())
    }

    pub fn write_run_finish(&mut self) -> io::Result<bool> {
        assert!(self.passed + self.failed + self.ignored + self.measured == self.total);

        let success = self.failed == 0;
        if !success {
            try!(self.write_failures());
        }

        try!(self.write_plain("\ntest result: "));
        if success {
            // There's no parallelism at this point so it's safe to use color
            try!(self.write_pretty("ok"));
        } else {
            try!(self.write_pretty("FAILED"));
        }
        let s = format!(". {} passed; {} failed; {} ignored; {} measured\n\n",
                        self.passed,
                        self.failed,
                        self.ignored,
                        self.measured);
        try!(self.write_plain(&s));
        Ok(success)
    }
}

// Format a number with thousands separators
fn fmt_thousands_sep(mut n: usize, sep: char) -> String {
    use std::fmt::Write;
    let mut output = String::new();
    let mut trailing = false;
    for &pow in &[9, 6, 3, 0] {
        let base = 10_usize.pow(pow);
        if pow == 0 || trailing || n / base != 0 {
            if !trailing {
                output.write_fmt(format_args!("{}", n / base)).unwrap();
            } else {
                output.write_fmt(format_args!("{:03}", n / base)).unwrap();
            }
            if pow != 0 {
                output.push(sep);
            }
            trailing = true;
        }
        n %= base;
    }

    output
}

pub fn fmt_bench_samples(bs: &BenchSamples) -> String {
    use std::fmt::Write;
    let mut output = String::new();

    let median = bs.ns_iter_summ.median as usize;
    let deviation = (bs.ns_iter_summ.max - bs.ns_iter_summ.min) as usize;

    output.write_fmt(format_args!("{:>11} ns/iter (+/- {})",
                                fmt_thousands_sep(median, ','),
                                fmt_thousands_sep(deviation, ',')))
        .unwrap();
    if bs.mb_s != 0 {
        output.write_fmt(format_args!(" = {} MB/s", bs.mb_s)).unwrap();
    }
    output
}

// A simple console test runner
pub fn run_tests_console(opts: &TestOpts, tests: Vec<TestDescAndFn>) -> io::Result<bool> {

    fn callback<T: Write>(event: &TestEvent, st: &mut ConsoleTestState<T>) -> io::Result<()> {
        match (*event).clone() {
            TeFiltered(ref filtered_tests) => st.write_run_start(filtered_tests.len()),
            TeWait(ref test, padding) => st.write_test_start(test, padding),
            TeResult(test, result, _) => {
                try!(st.write_log(&test, &result));
                try!(st.write_result(&result));
                match result {
                    TrIgnored => st.ignored += 1,
                    TrBench(_) => st.measured += 1,
                }
                Ok(())
            }
        }
    }

    let mut st = try!(ConsoleTestState::new(opts));
    fn len_if_padded(t: &TestDescAndFn) -> usize {
        match t.testfn.padding() {
            PadOnRight => t.desc.name.len(),
        }
    }
    if let Some(t) = tests.iter().max_by_key(|t| len_if_padded(*t)) {
        let n = &t.desc.name;
        st.max_name_len = n.len();
    }
    try!(run_tests(opts, tests, |x| callback(&x, &mut st)));
    st.write_run_finish()
}

#[test]
fn should_sort_failures_before_printing_them() {
    let test_a = TestDesc {
        name: Cow::from("a"),
        ignore: false,
    };

    let test_b = TestDesc {
        name: Cow::from("b"),
        ignore: false,
    };

    let mut st = ConsoleTestState {
        log_out: None,
        out: Raw(Vec::new()),
        quiet: false,
        total: 0,
        passed: 0,
        failed: 0,
        ignored: 0,
        measured: 0,
        max_name_len: 10,
        failures: vec![(test_b, Vec::new()), (test_a, Vec::new())],
    };

    st.write_failures().unwrap();
    let s = match st.out {
        Raw(ref m) => String::from_utf8_lossy(&m[..]),
    };

    let apos = s.find("a").unwrap();
    let bpos = s.find("b").unwrap();
    assert!(apos < bpos);
}

#[derive(Clone)]
enum TestEvent {
    TeFiltered(Vec<TestDesc>),
    TeWait(TestDesc, NamePadding),
    TeResult(TestDesc, TestResult, Vec<u8>),
}

type MonitorMsg = (TestDesc, TestResult, Vec<u8>);


fn run_tests<F>(opts: &TestOpts, tests: Vec<TestDescAndFn>, mut callback: F) -> io::Result<()>
    where F: FnMut(TestEvent) -> io::Result<()>
{

    let filtered_tests = filter_tests(opts, tests);

    let filtered_descs = filtered_tests.iter()
        .map(|t| t.desc.clone())
        .collect();

    try!(callback(TeFiltered(filtered_descs)));

    let filtered_benchs_and_metrics = filtered_tests;

    // All benchmarks run at the end, in serial.
    // (this includes metric fns)
    for b in filtered_benchs_and_metrics {
        try!(callback(TeWait(b.desc.clone(), b.testfn.padding())));
        let (test, result, stdout) = run_test(opts, false, b);
        try!(callback(TeResult(test, result, stdout)));
    }
    Ok(())
}

fn filter_tests(opts: &TestOpts, tests: Vec<TestDescAndFn>) -> Vec<TestDescAndFn> {
    let mut filtered = tests;

    // Remove tests that don't match the test filter
    filtered = match opts.filter {
        None => filtered,
        Some(ref filter) => {
            filtered.into_iter()
                .filter(|test| test.desc.name.contains(&filter[..]))
                .collect()
        }
    };

    // Maybe pull out the ignored test and unignore them
    filtered = if !opts.run_ignored {
        filtered
    } else {
        fn filter(test: TestDescAndFn) -> Option<TestDescAndFn> {
            if test.desc.ignore {
                let TestDescAndFn { desc, testfn } = test;
                Some(TestDescAndFn {
                    desc: TestDesc { ignore: false, ..desc },
                    testfn: testfn,
                })
            } else {
                None
            }
        }
        filtered.into_iter().filter_map(filter).collect()
    };

    // Sort the tests alphabetically
    filtered.sort_by(|t1, t2| t1.desc.name.cmp(&t2.desc.name));

    filtered
}

fn run_test(_opts: &TestOpts, force_ignore: bool, test: TestDescAndFn) -> MonitorMsg {

    let TestDescAndFn { desc, testfn } = test;

    if force_ignore || desc.ignore {
        return (desc, TrIgnored, Vec::new());
    }

    match testfn {
        DynBenchFn(bencher) => {
            let bs = bench::benchmark(|harness| bencher.run(harness));
            return (desc, TrBench(bs), Vec::new());
        }
        StaticBenchFn(benchfn) => {
            let bs = bench::benchmark(|harness| benchfn(harness));
            return (desc, TrBench(bs), Vec::new());
        }
    }
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
// TODO -
