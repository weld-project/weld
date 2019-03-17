//! Target-specific information and querying.
//!
//! This module provides functionality for querying whether target-specific features are available
//! on the current platform.

use fnv;

use std::fmt;
use std::str;

use fnv::FnvHashSet;

use crate::error::*;

/// X86-specific feature list.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum X86Feature {
    AES,
    AVX,
    AVX2,
    CMov,
    FMA,
    SSE,
    SSE2,
    SSE3,
    SSE4_1,
    SSE4_2,
    SSSE3,
}

impl fmt::Display for X86Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = format!("{:?}", self);
        write!(f, "{}", s.to_lowercase().replace("_", "."))
    }
}

impl str::FromStr for X86Feature {
    type Err = WeldCompileError;
    fn from_str(s: &str) -> WeldResult<X86Feature> {
        use self::X86Feature::*;
        match s {
            "aes" => Ok(AES),
            "avx" => Ok(AVX),
            "avx2" => Ok(AVX2),
            "cmov" => Ok(CMov),
            "fma" => Ok(FMA),
            "sse" => Ok(SSE),
            "sse2" => Ok(SSE2),
            "sse3" => Ok(SSE3),
            "sse4.1" => Ok(SSE4_1),
            "sse4.2" => Ok(SSE4_2),
            "ssse3" => Ok(SSSE3),
            other => compile_err!("Unrecognized or unsupported x86 feature '{}'", other),
        }
    }
}

/// Specifies a target and its features.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetFeatures {
    /// Currently counts both x86 and x86_64.
    X86(FnvHashSet<X86Feature>),
    /// Unknown target.
    Unknown,
}

impl fmt::Display for TargetFeatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use self::TargetFeatures::*;
        match *self {
            X86(ref features) => {
                let features = features
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                write!(f, "target=x86, features=[{}]", features)
            }
            Unknown => write!(f, "target=unknown, features=[]"),
        }
    }
}

impl TargetFeatures {
    /// Checks whether this `TargetFeatures` supports the given x86 feature.
    ///
    /// Returns `false` if this is not an x86 target or the feature is not supported.
    pub fn x86_supports(&self, feature: X86Feature) -> bool {
        match *self {
            TargetFeatures::X86(ref features) if features.contains(&feature) => true,
            _ => false,
        }
    }
}

pub struct Target {
    pub cpu: String,
    pub features: TargetFeatures,
}

impl Target {
    /// Returns a new `TargetFeature` using an LLVM feature and CPU string.
    pub fn from_llvm_strings<T: AsRef<str>, U: AsRef<str>, V: AsRef<str>>(
        triple_string: T,
        cpu_string: U,
        feature_string: V,
    ) -> WeldResult<Target> {
        let cpu = cpu_string.as_ref().to_string();
        let features = if triple_string.as_ref().contains("x86") {
            let mut features = FnvHashSet::default();
            let feature_tokens = feature_string
                .as_ref()
                .split(',')
                .filter(|s| s.starts_with('+'));
            for feature_token in feature_tokens {
                // Since we use the LLVM string, just ignore errors. They indicate features that we
                // don't support/know about yet.
                if let Ok(feature) = feature_token.get(1..).unwrap().parse::<X86Feature>() {
                    features.insert(feature);
                } else {
                    trace!("Unrecognized x86 feature {}", feature_token);
                }
            }
            TargetFeatures::X86(features)
        } else {
            TargetFeatures::Unknown
        };

        let result = Target { cpu, features };
        Ok(result)
    }
}
