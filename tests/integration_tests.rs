use std::env;

extern crate weld;

use weld::llvm::*;
use weld::parser::*;
use weld::weld_run_free;
use weld::weld_rt_malloc;
use weld::weld_rt_realloc;
use weld::weld_rt_free;
use weld::weld_rt_get_errno;
use weld::weld_rt_set_errno;
use weld::new_merger;
use weld::get_merger_at_index;
use weld::free_merger;
use weld::my_id_public;
use weld::set_result;
use weld::get_result;
use weld::get_nworkers;
use weld::set_nworkers;
use weld::get_runid;
use weld::set_runid;
use weld::pl_start_loop;
use weld::execute;
use weld::new_vb;
use weld::new_piece;
use weld::cur_piece;
use weld::result_vb;
use weld::WeldRuntimeErrno;

/// Required to prevent symbols from being optimized out during compilation.
fn runtime_fns() {
    // Weld Runtime.
    println!("{:p}", weld_rt_malloc as *const ());
    println!("{:p}", weld_rt_realloc as *const ());
    println!("{:p}", weld_rt_free as *const ());
    println!("{:p}", weld_rt_get_errno as *const ());
    println!("{:p}", weld_rt_set_errno as *const ());
    // Parlib
    println!("{:p}", my_id_public as *const ());
    println!("{:p}", set_result as *const ());
    println!("{:p}", get_result as *const ());
    println!("{:p}", get_nworkers as *const ());
    println!("{:p}", set_nworkers as *const ());
    println!("{:p}", get_runid as *const ());
    println!("{:p}", set_runid as *const ());
    println!("{:p}", pl_start_loop as *const ());
    println!("{:p}", execute as *const ());
    println!("{:p}", new_vb as *const ());
    println!("{:p}", new_piece as *const ());
    println!("{:p}", cur_piece as *const ());
    println!("{:p}", result_vb as *const ());
    // Builders
    println!("{:p}", new_merger as *const ());
    println!("{:p}", get_merger_at_index as *const ());
    println!("{:p}", free_merger as *const ());
}

fn basic_program() {
    let code = "|| 40 + 2";

    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let inp = Box::new(WeldInputArgs {
        input: 0,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result = module.run(ptr) as *const i32;
    let result = unsafe { *result };
    assert_eq!(result, 42);
    weld_run_free(-1);
}

fn f64_cast() {
    let code = "|| f64(40 + 2)";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();

    let inp = Box::new(WeldInputArgs {
        input: 0,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result = module.run(ptr) as *const f64;
    let result = unsafe { *result };
    assert_eq!(result, 42.0);
    weld_run_free(-1);
}

fn i32_cast() {
    let code = "|| i32(0.251 * 4.0)";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let inp = Box::new(WeldInputArgs {
        input: 0,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result = module.run(ptr) as *const i32;
    let result = unsafe { *result };
    assert_eq!(result, 1);
    weld_run_free(-1);
}

fn program_with_args() {
    let code = "|x:i32| 40 + x";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let input: i32 = 2;
    let inp = Box::new(WeldInputArgs {
        input: &input as *const i32 as i64,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result = module.run(ptr) as *const i32;
    let result = unsafe { *result };
    assert_eq!(result, 42);
    weld_run_free(-1);
}

fn let_statement() {
    let code = "|x:i32| let y = 40 + x; y + 2";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let input: i32 = 2;
    let inp = Box::new(WeldInputArgs {
        input: &input as *const i32 as i64,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result = module.run(ptr) as *const i32;
    let result = unsafe { *result };
    assert_eq!(result, 44);
    weld_run_free(-1);
}

fn if_statement() {
    let code = "|x:i32| if(true, 3, 4)";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let input: i32 = 2;
    let inp = Box::new(WeldInputArgs {
        input: &input as *const i32 as i64,
        nworkers: 1,
        run_id: 0,
    });

    let ptr = Box::into_raw(inp) as i64;
    let result = module.run(ptr) as *const i32;
    let result = unsafe { *result };
    assert_eq!(result, 3);
    weld_run_free(-1);
}

fn comparison() {
    let code = "|x:i32| if(x>10, x, 10)";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let mut input: i32 = 2;
    let inp = Box::new(WeldInputArgs {
        input: &input as *const i32 as i64,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result = module.run(ptr) as *const i32;
    let result = unsafe { *result };
    assert_eq!(result, 10);
    weld_run_free(-1);

    input = 20;
    let result = module.run(ptr) as *const i32;
    let result = unsafe { *result };
    assert_eq!(result, 20);
    weld_run_free(-1);
}

fn simple_vector_lookup() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }
    #[allow(dead_code)]
    struct Args {
        x: Vec,
    }

    let code = "|x:vec[i32]| lookup(x, 3L)";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let input = [1, 2, 4, 5];
    let args = Args {
        x: Vec {
            data: &input as *const i32,
            len: 4,
        },
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const Args as i64,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const i32;
    let result = unsafe { (*result_raw).clone() };
    let output = input[3];
    assert_eq!(output, result);
    weld_run_free(-1);
}

fn simple_for_appender_loop() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }
    #[allow(dead_code)]
    struct Args {
        x: Vec,
        a: i32,
    }

    let code = "|x:vec[i32], a:i32| let b=a+1; map(x, |e| e+b)";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let input = [1, 2];
    let args = Args {
        x: Vec {
            data: &input as *const i32,
            len: 2,
        },
        a: 1,
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const Args as i64,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const Vec;
    let result = unsafe { (*result_raw).clone() };
    let output = [3, 4];
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
    }
    weld_run_free(-1);
}

fn simple_parallel_for_appender_loop() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct WeldVec {
        data: *const i32,
        len: i64,
    }
    #[derive(Clone)]
    #[allow(dead_code)]
    struct WeldVec64 {
        data: *const i64,
        len: i64,
    }
    let code = "|x:vec[i32]| result(for(x, appender[i64], |b,i,e| merge(b, i)))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let size: i32 = 10000;
    let input: Vec<i32> = vec![0; size as usize];
    let args = WeldVec {
        data: input.as_ptr() as *const i32,
        len: size as i64,
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const WeldVec as i64,
        nworkers: 4,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const WeldVec64;
    let result = unsafe { (*result_raw).clone() };
    assert_eq!(result.len as i32, size);
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, i as i64)
    }
    weld_run_free(-1);
}

fn complex_parallel_for_appender_loop() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct WeldVec {
        data: *const i32,
        len: i64,
    }
    #[derive(Clone)]
    #[allow(dead_code)]
    struct WeldVec64 {
        data: *const i64,
        len: i64,
    }
    let code = "|x:vec[i32]| let a=appender[i64]; let b=merge(a,0L); let r=for(x,b,|b,i,e| \
                let c=merge(b,1L); let d=for(x,c,|b,i,e| if(i<1L, merge(b,i), b)); merge(d, 2L)); \
                result(merge(r,3L))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let size: i32 = 3000;
    let input: Vec<i32> = vec![0; size as usize];
    let args = WeldVec {
        data: input.as_ptr() as *const i32,
        len: size as i64,
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const WeldVec as i64,
        nworkers: 4,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const WeldVec64;
    let result = unsafe { (*result_raw).clone() };
    assert_eq!(result.len as i32, size * 3 + 2);
    assert_eq!(unsafe { *result.data.offset(0) }, 0);
    assert_eq!(unsafe { *result.data.offset((size * 3 + 1) as isize) }, 3);
    for i in 0..(size as isize) {
        assert_eq!(unsafe { *result.data.offset(i * 3 + 1) }, 1);
        assert_eq!(unsafe { *result.data.offset(i * 3 + 2) }, 0);
        assert_eq!(unsafe { *result.data.offset(i * 3 + 3) }, 2)
    }
    weld_run_free(-1);
}

fn simple_for_merger_loop() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }
    #[allow(dead_code)]
    struct Args {
        x: Vec,
        a: i32,
    }

    let code = "|x:vec[i32], a:i32| result(for(x, merger[i32,+], |b,i,e| merge(b, e+a)))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let input = [1, 2, 3, 4, 5];
    let args = Args {
        x: Vec {
            data: &input as *const i32,
            len: 5,
        },
        a: 1,
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const Args as i64,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const i32;
    let result = unsafe { (*result_raw).clone() };
    let output = 20;
    assert_eq!(result, output);
    weld_run_free(-1);
}

fn simple_for_vecmerger_loop() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }
    let code = "|x:vec[i32]| result(for(x, vecmerger[i32,+](x), |b,i,e| b))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let input = [1, 1, 1, 1, 1];
    let args = Vec {
        data: &input as *const i32,
        len: input.len() as i64,
    };
    let inp = Box::new(WeldInputArgs {
        input: &args as *const Vec as i64,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const Vec;
    let result = unsafe { (*result_raw).clone() };
    assert_eq!(result.len, input.len() as i64);
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, input[i as usize]);
    }
    weld_run_free(-1);
}

fn simple_for_vecmerger_loop_2() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }

    let code = "|x:vec[i32]| result(for(x, vecmerger[i32,+](x), |b,i,e| merge(b, {i,e*7})))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let input = [1, 1, 1, 1, 1];
    let args = Vec {
        data: &input as *const i32,
        len: input.len() as i64,
    };
    let inp = Box::new(WeldInputArgs {
        input: &args as *const Vec as i64,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const Vec;
    let result = unsafe { (*result_raw).clone() };
    assert_eq!(result.len, input.len() as i64);
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) },
                   input[i as usize] + input[i as usize] * 7);
    }
    weld_run_free(-1);
}

fn parallel_for_vecmerger_loop() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }

    let code = "|x:vec[i32]| result(for(x, vecmerger[i32,+](x), |b,i,e| merge(b, {i,e*7})))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let input = [1; 4096];
    let args = Vec {
        data: &input as *const i32,
        len: input.len() as i64,
    };
    let inp = Box::new(WeldInputArgs {
        input: &args as *const Vec as i64,
        nworkers: 4,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const Vec;
    let result = unsafe { (*result_raw).clone() };
    assert_eq!(result.len, input.len() as i64);
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) },
                   input[i as usize] + input[i as usize] * 7);
    }
    weld_run_free(-1);
}

fn simple_for_dictmerger_loop() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }
    #[allow(dead_code)]
    struct Pair {
        ele1: i32,
        ele2: i32,
    }
    #[derive(Clone)]
    #[allow(dead_code)]
    struct VecPrime {
        data: *const Pair,
        len: i64,
    }
    #[allow(dead_code)]
    struct Args {
        x: Vec,
        y: Vec,
    }

    let code = "|x:vec[i32], y:vec[i32]| tovec(result(for(zip(x,y), dictmerger[i32,i32,+], \
                |b,i,e| merge(b, e))))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let keys = [1, 2, 2, 1, 3];
    let values = [2, 3, 4, 2, 1];
    let args = Args {
        x: Vec {
            data: &keys as *const i32,
            len: 5,
        },
        y: Vec {
            data: &values as *const i32,
            len: 5,
        },
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const Args as i64,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const VecPrime;
    let result = unsafe { (*result_raw).clone() };
    let output_keys = [1, 2, 3];
    let output_values = [4, 7, 1];
    for i in 0..(output_keys.len() as isize) {
        let mut success = false;
        let key = unsafe { (*result.data.offset(i)).ele1 };
        let value = unsafe { (*result.data.offset(i)).ele2 };
        for j in 0..(output_keys.len()) {
            if output_keys[j] == key {
                if output_values[j] == value {
                    success = true;
                }
            }
        }
        assert_eq!(success, true);
    }
    assert_eq!(result.len, output_keys.len() as i64);
    weld_run_free(-1);
}

fn simple_parallel_for_dictmerger_loop() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }
    #[allow(dead_code)]
    struct Pair {
        ele1: i32,
        ele2: i32,
    }
    #[derive(Clone)]
    #[allow(dead_code)]
    struct VecPrime {
        data: *const Pair,
        len: i64,
    }
    #[allow(dead_code)]
    struct Args {
        x: Vec,
        y: Vec,
    }

    let code = "|x:vec[i32], y:vec[i32]| tovec(result(for(zip(x,y), dictmerger[i32,i32,+], \
                |b,i,e| merge(b, e))))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    const DICT_SIZE: usize = 8192;
    let mut keys = [0; DICT_SIZE];
    let mut values = [0; DICT_SIZE];
    for i in 0..DICT_SIZE {
        keys[i] = i as i32;
        values[i] = i as i32;
    }
    let args = Args {
        x: Vec {
            data: &keys as *const i32,
            len: DICT_SIZE as i64,
        },
        y: Vec {
            data: &values as *const i32,
            len: DICT_SIZE as i64,
        },
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const Args as i64,
        nworkers: 4,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const VecPrime;
    let result = unsafe { (*result_raw).clone() };
    let output_keys = keys;
    let output_values = values;
    for i in 0..(output_keys.len() as isize) {
        let mut success = false;
        let key = unsafe { (*result.data.offset(i)).ele1 };
        let value = unsafe { (*result.data.offset(i)).ele2 };
        for j in 0..(output_keys.len()) {
            if output_keys[j] == key {
                if output_values[j] == value {
                    success = true;
                }
            }
        }
        assert_eq!(success, true);
    }
    assert_eq!(result.len, output_keys.len() as i64);
    weld_run_free(-1);
}

fn simple_dict_lookup() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }
    #[allow(dead_code)]
    struct Args {
        x: Vec,
        y: Vec,
    }

    let code = "|x:vec[i32], y:vec[i32]| let a = result(for(zip(x,y), dictmerger[i32,i32,+], \
                |b,i,e| merge(b, e))); lookup(a, 1)";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let keys = [1, 2, 2, 1, 3];
    let values = [2, 3, 4, 2, 1];
    let args = Args {
        x: Vec {
            data: &keys as *const i32,
            len: 5,
        },
        y: Vec {
            data: &values as *const i32,
            len: 5,
        },
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const Args as i64,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const i32;
    let result = unsafe { (*result_raw).clone() };
    let output = 4;
    assert_eq!(output, result);
    weld_run_free(-1);
}

fn simple_length() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }
    #[allow(dead_code)]
    struct Args {
        x: Vec,
    }

    let code = "|x:vec[i32]| len(x)";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let data = [2, 3, 4, 2, 1];
    let args = Args {
        x: Vec {
            data: &data as *const i32,
            len: 5,
        },
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const Args as i64,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const i32;
    let result = unsafe { (*result_raw).clone() };
    let output = 5;
    assert_eq!(output, result);
    weld_run_free(-1);
}

fn filter_length() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }
    #[allow(dead_code)]
    struct Args {
        x: Vec,
    }

    let code = "|x:vec[i32]| len(filter(x, |i| i < 4 && i > 1))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let data = [2, 3, 4, 2, 1];
    let args = Args {
        x: Vec {
            data: &data as *const i32,
            len: 5,
        },
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const Args as i64,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const i32;
    let result = unsafe { (*result_raw).clone() };
    let output = 3;
    assert_eq!(output, result);
    weld_run_free(-1);
}

fn flat_map_length() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }
    #[allow(dead_code)]
    struct Args {
        x: Vec,
    }

    let code = "|x:vec[i32]| len(flatten(map(x, |i:i32| x)))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();

    let data = [2, 3, 4, 2, 1];
    let args = Args {
        x: Vec {
            data: &data as *const i32,
            len: 5,
        },
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const Args as i64,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const i32;
    let result = unsafe { (*result_raw).clone() };
    let output = 25;
    assert_eq!(output, result);
    weld_run_free(-1);
}

fn if_for_loop() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }
    #[allow(dead_code)]
    struct Args {
        x: Vec,
        a: i32,
    }

    let code = "|x:vec[i32], a:i32| if(a > 5, map(x, |e| e+1), map(x, |e| e+2))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let input = [1, 2];

    let args = Args {
        x: Vec {
            data: &input as *const i32,
            len: 2,
        },
        a: 1,
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const Args as i64,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const Vec;
    let result = unsafe { (*result_raw).clone() };
    let output = [3, 4];
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
    }
    weld_run_free(-1);

    let args = Args {
        x: Vec {
            data: &input as *const i32,
            len: 2,
        },
        a: 6,
    };
    let inp = Box::new(WeldInputArgs {
        input: &args as *const Args as i64,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const Vec;
    let result = unsafe { (*result_raw).clone() };
    let output = [2, 3];
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
    }
    weld_run_free(-1);
}

fn map_zip_loop() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }
    #[allow(dead_code)]
    struct Args {
        x: Vec,
        y: Vec,
    }

    let code = "|x:vec[i32], y:vec[i32]| map(zip(x,y), |e| e.$0 + e.$1)";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let x = [1, 2, 3, 4];
    let y = [5, 6, 7, 8];
    let args = Args {
        x: Vec {
            data: &x as *const i32,
            len: 4,
        },
        y: Vec {
            data: &y as *const i32,
            len: 4,
        },
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const Args as i64,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const Vec;
    let result = unsafe { (*result_raw).clone() };
    let output = [6, 8, 10, 12];
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
    }
    weld_run_free(-1);
}

fn iters_for_loop() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }
    #[allow(dead_code)]
    struct Args {
        x: Vec,
        y: Vec,
    }

    let code = "|x:vec[i32], y:vec[i32]| result(for(zip(iter(x,0L,4L,2L), y), appender, |b,i,e| \
                merge(b,e.$0+e.$1)))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let x = [1, 2, 3, 4];
    let y = [5, 6];
    let args = Args {
        x: Vec {
            data: &x as *const i32,
            len: 4,
        },
        y: Vec {
            data: &y as *const i32,
            len: 2,
        },
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const Args as i64,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const Vec;
    let result = unsafe { (*result_raw).clone() };
    let output = [6, 9];
    for i in 0..(result.len as isize) {
        assert_eq!(unsafe { *result.data.offset(i) }, output[i as usize])
    }
    weld_run_free(-1);
}

fn serial_parlib_test() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct WeldVec {
        data: *const i32,
        len: i64,
    }
    let code = "|x:vec[i32]| result(for(x, merger[i32,+], |b,i,e| merge(b, e)))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let size: i32 = 10000;
    let input: Vec<i32> = vec![1; size as usize];
    let args = WeldVec {
        data: input.as_ptr() as *const i32,
        len: size as i64,
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const WeldVec as i64,
        nworkers: 1,
        run_id: 0,
    });
    let ptr = Box::into_raw(inp) as i64;

    let result_raw = module.run(ptr) as *const i32;
    let result = unsafe { (*result_raw).clone() };
    assert_eq!(result, size);
    weld_run_free(-1);
}

fn iters_outofbounds_error_test() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }
    #[allow(dead_code)]
    struct Args {
        x: Vec,
    }

    let code = "|x:vec[i32]| result(for(iter(x,0L,20000L,1L), appender, |b,i,e| merge(b,e+1)))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    let x = [4; 1000 as usize];
    let args = Args {
        x: Vec {
            data: &x as *const i32,
            len: x.len() as i64,
        },
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const Args as i64,
        nworkers: 4,
        run_id: 5, // this test needs a unique run ID so we don't reset accidentally.
    });
    let ptr = Box::into_raw(inp) as i64;
    module.run(ptr) as *const i32;

    // Get the error back for the run ID we used.
    let errno = weld_rt_get_errno(5);
    assert_eq!(errno, WeldRuntimeErrno::BadIteratorLength);
    weld_run_free(5);
}

fn outofmemory_error_test() {
    #[derive(Clone)]
    #[allow(dead_code)]
    struct Vec {
        data: *const i32,
        len: i64,
    }
    #[allow(dead_code)]
    struct Args {
        x: Vec,
    }

    // TODO(shoumik): This test (and all the other tests) can be made more robust
    // by using the API directly, since that will test how users actually use Weld.
    // What we do here uses unstable APIs and could cause tests to break in the
    // future.
    //
    // This test tests the case where the default memory limit (1GB) is exceeded.
    let code = "|x:vec[i32]| result(for(x, appender, |b,i,e| merge(b,e+1)))";
    let module = compile_program(&parse_program(code).unwrap()).unwrap();
    // 1GB of data; the appender will allocate at least this much,
    // exceeding the 1GB default limit.
    let x = vec![4; 1000000000 / 4 as usize];
    let args = Args {
        x: Vec {
            data: x.as_ptr() as *const i32,
            len: x.len() as i64,
        },
    };

    let inp = Box::new(WeldInputArgs {
        input: &args as *const Args as i64,
        nworkers: 4,
        run_id: 6, // this test needs a unique run ID so we don't reset accidentally.
    });
    let ptr = Box::into_raw(inp) as i64;
    module.run(ptr) as *const i32;

    // Get the error back for the run ID we used.
    let errno = weld_rt_get_errno(6);
    assert_eq!(errno, WeldRuntimeErrno::OutOfMemory);
    weld_run_free(6);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let tests: Vec<(&str, fn())> =
        vec![("runtime_fns", runtime_fns),
             ("basic_program", basic_program),
             ("f64_cast", f64_cast),
             ("i32_cast", i32_cast),
             ("program_with_args", program_with_args),
             ("let_statement", let_statement),
             ("if_statement", if_statement),
             ("comparison", comparison),
             ("simple_vector_lookup", simple_vector_lookup),
             ("simple_for_appender_loop", simple_for_appender_loop),
             ("simple_parallel_for_appender_loop", simple_parallel_for_appender_loop),
             ("complex_parallel_for_appender_loop", complex_parallel_for_appender_loop),
             ("simple_for_merger_loop", simple_for_merger_loop),
             ("simple_for_vecmerger_loop", simple_for_vecmerger_loop),
             ("simple_for_vecmerger_loop_2", simple_for_vecmerger_loop_2),
             ("parallel_for_vecmerger_loop", parallel_for_vecmerger_loop),
             ("simple_for_dictmerger_loop", simple_for_dictmerger_loop),
             ("simple_parallel_for_dictmerger_loop", simple_parallel_for_dictmerger_loop),
             ("simple_dict_lookup", simple_dict_lookup),
             ("simple_length", simple_length),
             ("filter_length", filter_length),
             ("flat_map_length", flat_map_length),
             ("if_for_loop", if_for_loop),
             ("map_zip_loop", map_zip_loop),
             ("iters_for_loop", iters_for_loop),
             ("iters_outofbounds_error_test", iters_outofbounds_error_test),
             ("outofmemory_error_test", outofmemory_error_test),
             ("serial_parlib_test", serial_parlib_test)];

    println!("");
    println!("running tests");
    let mut passed = 0;
    for t in tests.iter() {
        match t.0 {
            // don't run this, they exist only to make sure functions don't get optimized out
            "runtime_fns" => {}
            _ => {
                if args.len() > 1 {
                    if !t.0.contains(args[1].as_str()) {
                        continue;
                    }
                }
                print!("{} ... ", t.0);
                t.1();
                println!("\x1b[0;32mok\x1b[0m");
                passed += 1;
            }
        }
    }

    println!("");
    println!("test result: \x1b[0;32mok\x1b[0m. {} passed; 0 failed; {} ignored; 0 measured",
             passed,
             tests.len() - passed - 1); // - 1 to ignore runtime_fns
    println!("");
}
