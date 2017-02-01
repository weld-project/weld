//! Benchmarks for Weld.
//!
//! To use this utility, add a new benchmarking function below. Wrap the part
//! you wish to measure in call to `bench.iter`. To register the benchmark,
//! add it to the dictionary in `registered_benchmarks`. The string name chosen
//! for the benchmark is the target name that will appear in the output file.
use std::collections::HashMap;

extern crate weld;
extern crate easy_ll;

use bencher;

use self::weld::*;
use bencher::Bencher;

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
    let x: Vec<i32> = vec![4; size as usize];
    let y: Vec<i32> = vec![5; size as usize];

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

    // TODO(shoumik): Free memory.

    // Run once to check for correctness.
    let result_raw = module.run(&args as *const Args as i64, 1) as *const WeldVec;
    let result = unsafe { (*result_raw).clone() };
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, x[0] + y[0]);
    }

    bench.iter(|| module.run(&args as *const Args as i64, 1));
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
    let x: Vec<i32> = vec![4; size as usize];

    let args = Args {
        x: WeldVec {
            data: x.as_ptr() as *const i32,
            len: size,
        },
    };

    // Check correctness.
    let expect = x[0] * 4 * (size as i32);
    let result_raw = module.run(&args as *const Args as i64, 1) as *const i32;
    let result = unsafe { *result_raw.clone() };
    assert_eq!(expect, result);

    bench.iter(|| module.run(&args as *const Args as i64, 1))
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
        let result_raw = module.run(&args as *const Args as i64, 1) as *const i32;
        let result = unsafe { *result_raw.clone() };
        assert_eq!(result, expect as i32);
    })
}

/// Register functions that can be run with the benchmarking suite here.
pub fn registered_benchmarks() -> HashMap<String, fn(&mut bencher::Bencher)> {
    let mut benchmarks_all: HashMap<String, fn(&mut bencher::Bencher)> = HashMap::new();
    benchmarks_all.insert("bench_integer_vector_sum".to_string(),
                          bench_integer_vector_sum);
    benchmarks_all.insert("bench_integer_map_reduce".to_string(),
                          bench_integer_map_reduce);
    benchmarks_all.insert("bench_tpch_q6".to_string(), bench_tpch_q6);
    benchmarks_all
}
