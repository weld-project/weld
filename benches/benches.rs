use std::collections::HashMap;
use std::collections::HashSet;

extern crate weld;
extern crate getopts;
extern crate easy_ll;

#[macro_use]
mod bencher;

use bencher::Bencher;
use weld::*;

use getopts::Options;

#[derive(Clone)]
#[allow(dead_code)]
struct WeldVec {
    data: *const i32,
    len: i64,
}

/// Compiles a string into an LLVM module.
fn compile(code: &str) -> easy_ll::CompiledModule {
    let code = code.trim();
    let module = llvm::compile_program(&parser::parse_program(code).unwrap());
    module.unwrap()
}

fn bench_integer_vector_sum(bench: &mut Bencher) {

    #[allow(dead_code)]
    struct Args {
        x: WeldVec,
        y: WeldVec,
    }

    let code = "|x:vec[i32],y:vec[i32]| map(zip(x,y), |e| e.$0 + e.$1)";
    let module = compile(code);

    let size: i64 = 10000000;
    let x = vec![4; size as usize];
    let y = vec![5; size as usize];

    let args = Args {
        x: WeldVec {
            data: x.as_ptr() as *const i32,
            len: size,
        },
        y: WeldVec {
            data: y.as_ptr() as *const i32,
            len: size,
        },
    };

    // Run the module.
    // TODO(shoumik): How to free this memory?
    bench.iter(|| module.run(&args as *const Args as i64))
}

fn bench_integer_map_reduce(bench: &mut Bencher) {

    #[allow(dead_code)]
    struct Args {
        x: WeldVec,
    }

    let code = "|x:vec[i32]| result(for(map(x, |e| e * 4), merger[i32,+], |b,i,e| \
                merge(b, e)))";
    let module = compile(code);

    let size: i64 = 10000000;
    let x = vec![4; size as usize];

    let args = Args {
        x: WeldVec {
            data: x.as_ptr() as *const i32,
            len: size,
        },
    };

    // Run the module.
    // TODO(shoumik): How to free this memory?
    bench.iter(|| module.run(&args as *const Args as i64))
}

fn bench_tpch_q6(bench: &mut Bencher) {
    let code = include_str!("benchmarks/tpch/q6.weld");

    #[allow(dead_code)]
    struct Args {
        l_shipdate: WeldVec,
        l_discount: WeldVec,
        l_quantity: WeldVec,
        l_ep: WeldVec,
    }

    // Setup the arguments.
    let size: i64 = 500;
    let l_shipdate: Vec<i32> = vec![19940505; size as usize];
    let l_discount: Vec<i32> = vec![6; size as usize];
    let l_quantity: Vec<i32> = vec![20; size as usize];
    let l_ep: Vec<i32> = vec![100; size as usize];

    // All predicates pass
    let expect = l_ep[0] * l_discount[0] * (size as i32);

    let args = Args {
        l_shipdate: WeldVec {
            data: l_shipdate.as_ptr() as *const i32,
            len: size,
        },
        l_discount: WeldVec {
            data: l_discount.as_ptr() as *const i32,
            len: size,
        },
        l_quantity: WeldVec {
            data: l_quantity.as_ptr() as *const i32,
            len: size,
        },
        l_ep: WeldVec {
            data: l_ep.as_ptr() as *const i32,
            len: size,
        },
    };
    let module = compile(code);
    bench.iter(|| {
        let result_raw = module.run(&args as *const Args as i64) as *const i32;
        let result = unsafe { *result_raw.clone() };
        assert_eq!(result, expect as i32);
    })
}

fn print_usage(benches: Vec<String>, opts: &Options) {
    let brief = format!("Usage: Call using bench.sh");
    print!("{}", opts.usage(&brief));
    println!("Benchmarks available:");
    for bench in benches.iter() {
        println!("\t{}", bench);
    }
}

fn run_benchmarks(filename: String, benches: HashMap<String, fn(&mut bencher::Bencher)>) {
    use bencher::bench::benchmark;
    use bencher::BenchSamples;

    for (bench_name, bench_fn) in benches.iter() {
        println!("{}", bench_name);
    }
}


fn main() {
    use std::env;

    // Register new benchmarks here.
    let mut benchmarks_all: HashMap<String, fn(&mut bencher::Bencher)> = HashMap::new();
    benchmarks_all.insert("bench_integer_vector_sum".to_string(),
                          bench_integer_vector_sum);
    benchmarks_all.insert("bench_integer_map_reduce".to_string(),
                          bench_integer_map_reduce);

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

    let targets = matches.opt_str("t").unwrap_or("".to_string());

    if let Some(target_str) = matches.opt_str("t") {
        let targets: HashSet<_> = targets.split(",").map(|s| String::from(s.trim())).collect();
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
