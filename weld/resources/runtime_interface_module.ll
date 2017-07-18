; Simple LLVM module containing wrappers for the functions in libweldrt, so that we can call them
; from weld after the library is dynamically linked into the program through easy_ll.

; Declarations of functions in libweldrt.
declare void @weld_rt_run_free(i64 %run_id);
declare i64 @weld_rt_memory_usage(i64 %run_id);

; A dummy run function to make easy_ll compile this.
define i64 @run(i64) {
    ret i64 0
}

define i64 @rt_run_free(i64 %run_id) {
    call void @weld_rt_run_free(i64 %run_id)
    ret i64 0
}

define i64 @rt_memory_usage(i64 %run_id) {
    %res = call i64 @weld_rt_memory_usage(i64 %run_id)
    ret i64 %res
}
