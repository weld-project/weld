use libc::{c_void, int64_t, int32_t, size_t};

#[allow(non_camel_case_types)]
type work_t = c_void;

#[allow(non_camel_case_types)]
type vec_piece = c_void;

#[repr(C)]
pub struct vec_output {
    data: *mut c_void,
    size: i64,
}

#[link(name="weldrt", kind="static")]
extern "C" {
    #[no_mangle]
    pub fn weld_runtime_init();
    #[no_mangle]
    pub fn weld_rt_thread_id() -> int32_t;
    #[no_mangle]
    pub fn weld_rt_abort_thread();
    #[no_mangle]
    pub fn weld_rt_get_nworkers() -> int32_t;
    #[no_mangle]
    pub fn weld_rt_get_run_id() -> int64_t;
    #[no_mangle]
    pub fn weld_rt_start_loop(w: *mut work_t,
                              body_data: *mut c_void,
                              cont_data: *mut c_void,
                              body: extern "C" fn(*mut work_t),
                              cont: extern "C" fn(*mut work_t),
                              lower: int64_t,
                              upper: int64_t,
                              grain_size: int32_t);
    #[no_mangle]
    pub fn weld_rt_set_result(res: *mut c_void);
    #[no_mangle]
    pub fn weld_rt_new_vb(elem_size: int64_t, starting_cap: int64_t, fixed_size: int32_t) -> *mut c_void;
    #[no_mangle]
    pub fn weld_rt_new_vb_piece(v: *mut c_void, w: *mut work_t);
    #[no_mangle]
    pub fn weld_rt_cur_vb_piece(v: *mut c_void, my_id: int32_t) -> *mut vec_piece;
    #[no_mangle]
    pub fn weld_rt_set_vb_offset_if_fixed(v: *mut c_void, offset: int64_t);
    #[no_mangle]
    pub fn weld_rt_result_vb(v: *mut c_void) -> vec_output;
    #[no_mangle]
    pub fn weld_rt_new_merger(size: int64_t, nworkers: int32_t) -> *mut c_void;
    #[no_mangle]
    pub fn weld_rt_get_merger_at_index(m: *mut c_void, size: int64_t, i: int32_t) -> *mut c_void;
    #[no_mangle]
    pub fn weld_rt_free_merger(m: *mut c_void);
    #[no_mangle]
    pub fn weld_run_begin(run: extern "C" fn(*mut work_t), mem_limit: int64_t, n_workers: int32_t) -> int64_t;
    #[no_mangle]
    pub fn weld_run_get_result(run_id: int64_t) -> *mut c_void;
    #[no_mangle]
    pub fn weld_run_dispose(run_id: int64_t);
    #[no_mangle]
    pub fn weld_run_malloc(run_id: int64_t, size: size_t) -> *mut c_void;
    #[no_mangle]
    pub fn weld_run_realloc(run_id: int64_t, data: *mut c_void, size: size_t) -> *mut c_void;
    #[no_mangle]
    pub fn weld_run_free(run_id: int64_t, data: *mut c_void);
    #[no_mangle]
    pub fn weld_run_memory_usage(run_id: int64_t) -> int64_t;
    #[no_mangle]
    pub fn weld_run_get_errno(run_id: int64_t) -> int64_t;
    #[no_mangle]
    pub fn weld_run_set_errno(run_id: int64_t, err: int64_t);
    #[no_mangle]
    pub fn weld_rt_dict_new(key_size: int32_t,
        keys_eq: extern "C" fn(*mut c_void, *mut c_void) -> int32_t,
        merge_new_val: extern "C" fn(int32_t, *mut c_void, *mut c_void),
        merge_vals_finalize: extern "C" fn(int32_t, *mut c_void, *mut c_void),
        metadata: *mut c_void, val_size: int32_t, to_array_true_val_size: int32_t,
        max_local_bytes: int64_t, capacity: int64_t) -> *mut c_void;
    #[no_mangle]
    pub fn weld_rt_dict_new_finalized(key_size: int32_t,
        keys_eq: extern "C" fn(*mut c_void, *mut c_void) -> int32_t,
        merge_new_val: extern "C" fn(int32_t, *mut c_void, *mut c_void),
        merge_vals_finalize: extern "C" fn(int32_t, *mut c_void, *mut c_void),
        metadata: *mut c_void, val_size: int32_t, to_array_true_val_size: int32_t,
        max_local_bytes: int64_t, capacity: int64_t) -> *mut c_void;
    #[no_mangle]
    pub fn weld_rt_dict_lookup(d: *mut c_void, hash: int32_t, key: *mut c_void) -> *mut c_void;
    #[no_mangle]
    pub fn weld_rt_dict_merge(d: *mut c_void, hash: int32_t, key: *mut c_void, value: *mut c_void);
    #[no_mangle]
    pub fn weld_rt_dict_finalize(d: *mut c_void);
    #[no_mangle]
    pub fn weld_rt_dict_to_array(d: *mut c_void, value_offset_in_struct: int32_t,
        struct_size: int32_t) -> *mut c_void;
    #[no_mangle]
    pub fn weld_rt_dict_size(d: *mut c_void) -> int64_t;
    #[no_mangle]
    pub fn weld_rt_dict_serialize(d: *mut c_void,
                                  buf: *mut c_void,
                                  has_pointer: int32_t,
                                  key_ser: extern "C" fn(*mut c_void, *mut c_void),
                                  val_ser: extern "C" fn(*mut c_void, *mut c_void));
    #[no_mangle]
    pub fn weld_rt_dict_free(d: *mut c_void);
    #[no_mangle]
    pub fn weld_rt_gb_new(key_size: int32_t,
        keys_eq: extern "C" fn(*mut c_void, *mut c_void) -> int32_t, val_size: int32_t,
        max_local_bytes: int64_t, capacity: int64_t) -> *mut c_void;
    #[no_mangle]
    pub fn weld_rt_gb_merge(b: *mut c_void, key: *mut c_void, hash: int32_t, value: *mut c_void);
    #[no_mangle]
    pub fn weld_rt_gb_result(b: *mut c_void) -> *mut c_void;
    #[no_mangle]
    pub fn weld_rt_gb_free(gb: *mut c_void);
}
