//! Utility struct for measuring compilation time.

use time;

use self::time::Duration;

/// Tracks various compile-time statistics throughout the compiler.
pub struct CompilationStats {
    /// Running times for various Weld compiler components.
    pub weld_times: Vec<(String, Duration)>,
    /// Running times for Weld optimization passes.
    pub pass_times: Vec<(String, Duration)>,
    /// Running times for various LLVM components.
    pub llvm_times: Vec<(String, Duration)>,
}

impl CompilationStats {
    pub fn new() -> CompilationStats {
        CompilationStats {
            weld_times: Vec::new(),
            pass_times: Vec::new(),
            llvm_times: Vec::new(),
        }
    }

    /// Formats a duration for printing the statistics, in terms of milliseconds and microseconds.
    fn format_time(duration: &Duration) -> f64 {
        if duration.num_milliseconds() == 0 {
            if let Some(v) = duration.num_microseconds() {
                (v as f64) / 1000.0
            } else {
                0.0
            }
        } else {
            duration.num_milliseconds() as f64
        }
    }

    /// Returns pretty-printed statistics stored in `self`.
    pub fn pretty_print(&self) -> String {
        let mut result = String::new();
        result.push_str("Weld Compiler:\n");
        let mut total = Duration::milliseconds(0);
        for &(ref name, ref dur) in self.weld_times.iter() {
            result.push_str(&format!(
                "\t{}: {:.3} ms\n",
                name,
                CompilationStats::format_time(dur)
            ));
            total = total + *dur;
        }
        result.push_str(&format!(
            "\t\x1b[0;32mWeld Compiler Total\x1b[0m {} ms\n",
            CompilationStats::format_time(&total)
        ));

        let mut total = Duration::milliseconds(0);
        result.push_str("Weld Optimization Passes:\n");
        for &(ref name, ref dur) in self.pass_times.iter() {
            result.push_str(&format!(
                "\t{}: {:.3} ms\n",
                name,
                CompilationStats::format_time(dur)
            ));
            total = total + *dur;
        }
        result.push_str(&format!(
            "\t\x1b[0;32mWeld Optimization Passes Total\x1b[0m {} ms\n",
            CompilationStats::format_time(&total)
        ));

        let mut total = Duration::milliseconds(0);
        result.push_str("LLVM:\n");
        for &(ref name, ref dur) in self.llvm_times.iter() {
            result.push_str(&format!(
                "\t{}: {:.3} ms\n",
                name,
                CompilationStats::format_time(dur)
            ));
            total = total + *dur;
        }
        result.push_str(&format!(
            "\t\x1b[0;32mLLVM Total\x1b[0m {} ms\n",
            CompilationStats::format_time(&total)
        ));

        result
    }
}
