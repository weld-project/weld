use std::collections::HashMap;
use std::collections::HashSet;

extern crate getopts;
extern crate csv;

mod bencher;
mod benches;

use getopts::Options;

fn print_usage(benches: Vec<String>, opts: &Options) {
    let brief = format!("Usage: Call using bench.sh");
    print!("{}", opts.usage(&brief));
    println!("Benchmarks available:");
    for bench in benches.iter() {
        println!("\t{}", bench);
    }
}

/// Formats a time in ns as a value in ms.
fn format_ns(x: f64) -> String {
    format!("{:.4}", x / 1000000_f64)
}

fn run_benchmarks(filename: String, benches: HashMap<String, fn(&mut bencher::Bencher)>) {
    use bencher::bench::benchmark;
    use bencher::stats::Summary;
    use csv;

    let mut wtr = csv::Writer::from_file(filename).unwrap();
    // Encode the CSV header.
    let result =
        wtr.encode(("name",
                    "config",
                    "mean(ms)",
                    "min(ms)",
                    "max(ms)",
                    "median(ms)",
                    "std_dev(ms)"));
    assert!(result.is_ok());
    for (bench_name, bench_fn) in benches.iter() {
        // Extract statistics.
        let Summary { min, max, mean, median, std_dev, .. } = benchmark(bench_fn).ns_iter_summ;
        let record = (bench_name,
                      "weld",
                      format_ns(mean),
                      format_ns(min),
                      format_ns(max),
                      format_ns(median),
                      format_ns(std_dev));
        let result = wtr.encode(record);
        assert!(result.is_ok());
    }
}


fn main() {
    use std::env;

    // Get the benchmarks.
    let benchmarks_all = benches::registered_benchmarks();

    let args: Vec<String> = env::args().collect();
    let mut opts = Options::new();
    opts.optopt("t", "targets", "Choose benchmarks to run", "TARGETS");
    opts.optopt("o", "", "Output file name", "FILE");
    opts.optflag("b", "bench", ""); // Needed for cargo bench
    let matches = match opts.parse(&args[0..]) {
        Ok(m) => m,
        Err(f) => panic!(f.to_string()),
    };


    let mut benches_final: HashMap<String, fn(&mut bencher::Bencher)> = HashMap::new();
    if let Some(target_str) = matches.opt_str("t") {
        let targets: HashSet<_> = target_str.split(",").map(|s| String::from(s.trim())).collect();
        for target in targets.iter() {
            if !benchmarks_all.contains_key(target) {
                println!("Benchmark not found: {}", target);
                print_usage(benchmarks_all.keys().map(|s| s.to_string()).collect(),
                            &opts);
                return;
            }
            benches_final.insert(target.clone(), *benchmarks_all.get(target).unwrap());
        }
    } else {
        // If no option was provided, run all the benchmarks.
        benches_final = benchmarks_all;
    }

    let filename = matches.opt_str("o").unwrap_or("bench.csv".to_string());
    run_benchmarks(filename, benches_final);
}
