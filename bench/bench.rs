#[macro_use]

extern crate bencher;
extern crate weld;
extern crate easy_ll;

use bencher::Bencher;

use weld::*;

#[derive(Clone)]
#[allow(dead_code)]
struct WeldVec {
    data: *const i32,
    len: i64,
}

fn compile(code: &str) -> easy_ll::CompiledModule {
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

// Add other benchmarks here.
benchmark_group!(benchmarks, bench_integer_vector_sum);
benchmark_main!(benchmarks);
