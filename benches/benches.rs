//! Benchmarks for Weld.
//!
//! To use this utility, add a new benchmarking function below. Wrap the part
//! you wish to measure in call to `bench.iter`. To register the benchmark,
//! add it to the dictionary in `registered_benchmarks`. The string name chosen
//! for the benchmark is the target name that will appear in the output file.

extern crate weld;
extern crate libc;

use bencher;
use std;

use self::weld::weld_print_function_pointers;
use self::weld::WeldRuntimeErrno;

use self::weld::{weld_conf_new, weld_conf_set, weld_conf_free};
use self::weld::{weld_error_code, weld_error_free};
use self::weld::{weld_value_new, weld_value_data, weld_value_free};
use self::weld::{weld_module_compile, weld_module_run, weld_module_free};
use self::weld::{WeldModule, WeldValue, WeldConf, WeldError};

use std::ffi::CString;
use self::libc::{c_char, c_void};

use std::collections::HashMap;

use bencher::Bencher;

#[derive(Clone)]
#[allow(dead_code)]
struct WeldVec<T> {
    data: *const T,
    len: i64,
}

/// Returns a configuration which uses several threads.
fn benchmark_conf() -> *mut WeldConf {
    let conf = weld_conf_new();
    let key = CString::new("weld.threads").unwrap().into_raw() as *const c_char;
    let value = CString::new("1").unwrap().into_raw() as *const c_char;
    unsafe { weld_conf_set(conf, key, value) };

    let key = CString::new("weld.memory.limit").unwrap().into_raw() as *const c_char;
    let value = CString::new("8589934592").unwrap().into_raw() as *const c_char;
    unsafe { weld_conf_set(conf, key, value) };
    conf
}

/// Returns a compiled, runnable Weld module.
unsafe fn compile_program(code: &str) -> Result<*mut WeldModule, ()> {
    let code = CString::new(code).unwrap();
    let conf = benchmark_conf();

    let mut err = std::ptr::null_mut();
    let module = weld_module_compile(code.into_raw() as *const c_char,
                                     conf,
                                     &mut err as *mut *mut WeldError);

    if weld_error_code(err) != WeldRuntimeErrno::Success {
        weld_conf_free(conf);
        return Err(());
    }
    weld_error_free(err);
    Ok(module)
}

/// Runs a `WeldModule` and returns either a `WeldValue` or `WeldError`, depending on whether the
/// run was successful.
unsafe fn run_module<T>(module: *mut WeldModule,
                        ptr: &T)
                        -> Result<*mut WeldValue, *mut WeldError> {
    let input_value = weld_value_new(ptr as *const _ as *const c_void);
    let conf = benchmark_conf();
    let mut err = std::ptr::null_mut();

    // Timing information per run for debugging.
    use std::time::Instant;
    let s = Instant::now();

    let ret_value = weld_module_run(module, conf, input_value, &mut err as *mut *mut WeldError);

    let dur = s.elapsed();
    let dur = dur.as_secs() * 1_000_000_000 + (dur.subsec_nanos() as u64);
    let dur = dur / 1_000_000;
    println!("{} ms", dur);

    // Free the input value wrapper.
    weld_value_free(input_value);
    weld_conf_free(conf);

    if weld_error_code(err) != WeldRuntimeErrno::Success {
        Err(err)
    } else {
        Ok(ret_value)
    }
}

fn bench_vector_sum(bench: &mut Bencher) {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
        y: WeldVec<i32>,
    }

    let code = include_str!("benchmarks/vector_sum.weld");

    // 100MB of data
    let data_size: usize = 2 << 26;
    let size: usize = data_size / std::mem::size_of::<i32>();

    let x: Vec<i32> = vec![4; size as usize];
    let y: Vec<i32> = vec![5; size as usize];

    let ref args = Args {
        x: WeldVec {
            data: x.as_ptr() as *const i32,
            len: size as i64,
        },
        y: WeldVec {
            data: y.as_ptr() as *const i32,
            len: size as i64,
        },
    };

    let module = unsafe { compile_program(code).unwrap() };

    // Run once to check correctness/warm up.
    let ret_value = unsafe { run_module(module, args).unwrap() };
    let data = unsafe { weld_value_data(ret_value) as *const WeldVec<i32> };
    let result = unsafe { (*data).clone() };

    assert_eq!(result.len, size as i64);
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) },
                   x[i as usize] + y[i as usize]);
    }
    unsafe { weld_value_free(ret_value) };

    bench.iter(|| {
        match unsafe { run_module(module, args) } {
            Ok(v) => unsafe { weld_value_free(v) },
            Err(e) => unsafe { weld_error_free(e) },
        }
    });

    unsafe { weld_module_free(module) };
}

fn bench_map_reduce(bench: &mut Bencher) {
    let code = include_str!("benchmarks/map_reduce.weld");

    // 100MB of data
    let data_size: usize = 2 << 26;
    let size: usize = data_size / std::mem::size_of::<i32>();

    let x: Vec<i32> = vec![4; size as usize];

    let ref args = WeldVec {
        data: x.as_ptr() as *const i32,
        len: size as i64,
    };

    let module = unsafe { compile_program(code).unwrap() };

    // Run once to check correctness/warm up.
    let ret_value = unsafe { run_module(module, args).unwrap() };
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };

    let expect = x[0] * 4 * (x.len() as i32);
    assert_eq!(result, expect);
    unsafe { weld_value_free(ret_value) };

    bench.iter(|| {
        match unsafe { run_module(module, args) } {
            Ok(v) => unsafe { weld_value_free(v) },
            Err(e) => unsafe { weld_error_free(e) },
        }
    });
}

fn bench_tpch_q1(bench: &mut Bencher) {
    let code = include_str!("benchmarks/tpch/q1.weld");

    #[allow(dead_code)]
    struct Args {
        l_returnflag: WeldVec<i8>,
        l_linestatus: WeldVec<i8>,
        l_quantity: WeldVec<f32>,
        l_ep: WeldVec<f32>,
        l_discount: WeldVec<f32>,
        l_shipdate: WeldVec<i32>,
        l_tax: WeldVec<f32>,
    }

    struct ReturnValue {
        l_returnflag: i8,
        l_linestatus: i8,
        sum_qty: f32,
        sum_base_price: f32,
        sum_disc_price: f32,
        sum_charge: f32,
        avg_qty: f32,
        avg_price: f32,
        count_order: i32,
    }

    // 100MB of data per column.
    let data_size: usize = 2 << 26;
    let size: usize = data_size / std::mem::size_of::<i32>();

    // Set values so all predicates pass.
    let l_returnflag: Vec<i8> = vec![1; size ];
    let l_linestatus: Vec<i8> = vec![1; size ];
    let l_quantity: Vec<f32> = vec![20.0; size ];
    let l_ep: Vec<f32> = vec![100.0; size];
    let l_discount: Vec<f32> = vec![6.0; size ];
    let l_shipdate: Vec<i32> = vec![19940505; size ];
    let l_tax: Vec<f32> = vec![6.0; size ];


    let ref args = Args {
        l_returnflag: WeldVec {
            data: l_returnflag.as_ptr() as *const i8,
            len: size as i64,
        },
        l_linestatus: WeldVec {
            data: l_linestatus.as_ptr() as *const i8,
            len: size as i64,
        },
        l_quantity: WeldVec {
            data: l_quantity.as_ptr() as *const f32,
            len: size as i64,
        },
        l_ep: WeldVec {
            data: l_ep.as_ptr() as *const f32,
            len: size as i64,
        },
        l_discount: WeldVec {
            data: l_discount.as_ptr() as *const f32,
            len: size as i64,
        },
        l_shipdate: WeldVec {
            data: l_shipdate.as_ptr() as *const i32,
            len: size as i64,
        },
        l_tax: WeldVec {
            data: l_tax.as_ptr() as *const f32,
            len: size as i64,
        },
    };

    let module = unsafe { compile_program(code).unwrap() };

    // Run once to check correctness/warm up.
    let ret_value = unsafe { run_module(module, args).unwrap() };
    let data = unsafe { weld_value_data(ret_value) as *const i32 };
    let result = unsafe { *data };

    // TODO(shoumik)
    // Since all predicates pass, this is the result
    // let expect = 0;
    // assert_eq!(result, expect);
    unsafe { weld_value_free(ret_value) };

    bench.iter(|| {
        match unsafe { run_module(module, args) } {
            Ok(v) => unsafe { weld_value_free(v) },
            Err(e) => unsafe { weld_error_free(e) },
        }
    });
}

fn bench_tpch_q6(bench: &mut Bencher) {
    let code = include_str!("benchmarks/tpch/q6.weld");

    #[allow(dead_code)]
    struct Args {
        l_shipdate: WeldVec<i32>,
        l_discount: WeldVec<f32>,
        l_quantity: WeldVec<f32>,
        l_ep: WeldVec<f32>,
    }

    // 100MB of data per column.
    let data_size: usize = 2 << 26;
    let size: usize = data_size / std::mem::size_of::<i32>();

    // Set values so all predicates pass.
    let l_shipdate: Vec<i32> = vec![19940505; size ];
    let l_discount: Vec<f32> = vec![6.0; size ];
    let l_quantity: Vec<f32> = vec![20.0; size ];
    let l_ep: Vec<f32> = vec![100.0; size];

    // Since all predicates pass, this is the result
    let expect = l_ep[0] * l_discount[0] * (size as f32);

    let ref args = Args {
        l_shipdate: WeldVec {
            data: l_shipdate.as_ptr() as *const i32,
            len: size as i64,
        },
        l_discount: WeldVec {
            data: l_discount.as_ptr() as *const f32,
            len: size as i64,
        },
        l_quantity: WeldVec {
            data: l_quantity.as_ptr() as *const f32,
            len: size as i64,
        },
        l_ep: WeldVec {
            data: l_ep.as_ptr() as *const f32,
            len: size as i64,
        },
    };

    let module = unsafe { compile_program(code).unwrap() };

    // Run once to check correctness/warm up.
    let ret_value = unsafe { run_module(module, args).unwrap() };
    let data = unsafe { weld_value_data(ret_value) as *const f32 };
    let result = unsafe { *data };

    assert_eq!(result, expect);
    unsafe { weld_value_free(ret_value) };

    bench.iter(|| {
        match unsafe { run_module(module, args) } {
            Ok(v) => unsafe { weld_value_free(v) },
            Err(e) => unsafe { weld_error_free(e) },
        }
    });
}



/// Register functions that can be run with the benchmarking suite here.
pub fn registered_benchmarks() -> HashMap<String, fn(&mut bencher::Bencher)> {

    weld_print_function_pointers();

    let mut benchmarks_all: HashMap<String, fn(&mut bencher::Bencher)> = HashMap::new();
    benchmarks_all.insert("bench_vector_sum".to_string(), bench_vector_sum);
    benchmarks_all.insert("bench_map_reduce".to_string(), bench_map_reduce);
    // benchmarks_all.insert("bench_tpch_q1".to_string(), bench_tpch_q1);
    // benchmarks_all.insert("bench_tpch_q6".to_string(), bench_tpch_q6);
    benchmarks_all
}
