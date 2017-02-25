
; Requires linking with libweldrt
declare void @weld_rt_run_free(i64 %run_id);

define i64 @run(i64 %run_id) {
    call void @weld_rt_run_free(i64 %run_id)
    ret i64 0
} 
