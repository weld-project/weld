//! Tests related to Weld's SIMD vectorizer transformation.

use weld;
use weld::WeldValue;

mod common;
use crate::common::*;

#[test]
fn simple_for_vectorizable_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
    }

    let code = "|x:vec[i32]| result(for(simditer(x), merger[i32,+], |b,i,e:simd[i32]| let a = broadcast(1); let a2 = a +
                    broadcast(1); merge(b, e+a2)))";
    let ref conf = default_conf();

    let size = 1000;
    let input_vec = vec![1 as i32; size as usize];
    let ref input_data = WeldVec::from(&input_vec);

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };
    let output = size * 3;
    assert_eq!(result, output);
}

#[test]
fn fringed_for_vectorizable_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
    }

    let code = "|x:vec[i32]|
	let b1 = for(
		simditer(x),
		merger[i32,+],
		|b,i,e:simd[i32]| let a = broadcast(1); let a2 = a + broadcast(1); merge(b, e+a2));
	result(for(fringeiter(x),
		b1,
		|b,i,e| let a = 1; let a2 = a + 1; merge(b, e+a2)
	))";

    let ref conf = default_conf();

    let size = 1002;
    let input_vec = vec![1 as i32; size as usize];
    let ref input_data = Args {
        x: WeldVec::from(&input_vec),
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };
    let output = size * 3;
    assert_eq!(result, output);
}

#[test]
fn fringed_for_vectorizable_loop_with_par() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
    }

    let code = "|x:vec[i32]|
	let b1 = for(
		simditer(x),
		merger[i32,+],
		|b,i,e:simd[i32]| let a = broadcast(1); let a2 = a + broadcast(1); merge(b, e+a2));
	result(for(fringeiter(x),
		b1,
		|b,i,e| let a = 1; let a2 = a + 1; merge(b, e+a2)
	))";

    let ref conf = many_threads_conf();

    // Large size to invoke parallel runtime + some fringing.
    let size = 10 * 1000 * 1000 + 123;
    let input_vec = vec![1 as i32; size as usize];
    let ref input_data = Args {
        x: WeldVec::from(&input_vec),
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };
    let output = size * 3;
    assert_eq!(result, output);
}

#[test]
fn for_predicated_vectorizable_loop() {
    #[allow(dead_code)]
    struct Args {
        x: WeldVec<i32>,
    }

    let code = "|x:vec[i32]|
	result(for(
			simditer(x:vec[i32]),
			merger[i32,+],
			|b:merger[i32,+],i:i64,e:simd[i32]|
			  merge(b:merger[i32,+],select(
				(e:simd[i32]>broadcast(0)),
				e:simd[i32],
				broadcast(0):simd[i32]
			  ))
	))
	";
    let ref conf = default_conf();

    let size = 1000;
    let input_vec = vec![1 as i32; size as usize];
    let ref input_data = Args {
        x: WeldVec::from(&input_vec),
    };

    let ret_value = compile_and_run(code, conf, input_data);
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };
    let output = size;
    assert_eq!(result, output);
}

fn check_result_and_free(ret_value: WeldValue, expected: i32) {
    let data = ret_value.data() as *const i32;
    let result = unsafe { (*data).clone() };
    assert_eq!(result, expected);
}

#[test]
fn predicate_if_iff_annotated() {
    #[allow(dead_code)]
    struct Args {
        v: WeldVec<i32>,
    }

    let input_vec = vec![-1, 2, 3, 4, 5];
    let ref input_data = Args {
        v: WeldVec::from(&input_vec),
    };

    let expected = 14;

    /* annotation true */
    let code = "|v:vec[i32]| result(for(v, merger[i32,+], |b,i,e| @(predicate:true)if(e>0, merge(b,e), b)))";
    let ref conf = default_conf();
    let ret_value = compile_and_run(code, conf, input_data);
    check_result_and_free(ret_value, expected);

    /* annotation false */
    let code = "|v:vec[i32]| result(for(v, merger[i32,+], |b,i,e| @(predicate:false)if(e>0, merge(b,e), b)))";
    let ref conf = default_conf();
    let ret_value = compile_and_run(code, conf, input_data);
    check_result_and_free(ret_value, expected);

    /* annotation missing */
    let code = "|v:vec[i32]| result(for(v, merger[i32,+], |b,i,e| if(e>0, merge(b,e), b)))";
    let ref conf = default_conf();
    let ret_value = compile_and_run(code, conf, input_data);
    check_result_and_free(ret_value, expected);
}
