
/// Defines a function called `$group_name` that returns the test description
/// values for the listed functions `$function`.
#[macro_export]
macro_rules! benchmark_group {
    ($group_name:ident, $($function:path),+) => {
        pub fn $group_name() -> ::std::vec::Vec<$crate::TestDescAndFn> {
            use $crate::{TestDescAndFn, TestFn, TestDesc};
            use std::borrow::Cow;
            let mut benches = ::std::vec::Vec::new();
            $(
                benches.push(TestDescAndFn {
                    desc: TestDesc {
                        name: Cow::from(stringify!($function)),
                        ignore: false,
                    },
                    testfn: TestFn::StaticBenchFn($function),
                });
            )+
            benches
        }
    }
}


/// Define a `fn main()` that will run all benchmarks defined by the groups
/// in `$group_name`.
///
/// The main function will read the first argument from the console and use
/// it to filter the benchmarks to run.
#[macro_export]
macro_rules! benchmark_main {
    ($($group_name:path),+) => {
        fn main() {
            use $crate::TestOpts;
            use $crate::run_tests_console;
            let mut test_opts = TestOpts::default();
            // check to see if we should filter:
            for arg in ::std::env::args().skip(1).filter(|arg| *arg != "--bench") {
                test_opts.filter = Some(arg);
                break;
            }
            let mut benches = Vec::new();
            $(
                benches.extend($group_name());
            )+
            run_tests_console(&test_opts, benches).unwrap();
        }
    }
}
