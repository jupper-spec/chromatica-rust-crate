use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
 
// ----------------------
// RANDOM SYSTEM
// ----------------------
//
// Uses a thread-local SmallRng so:
//   - No cross-thread data races
//   - No global atomic TOCTOU bugs
//   - SmallRng is fast and good enough for non-crypto use
//
// `srand` reseeds only the calling thread's RNG, making results
// reproducible on that thread without interfering with others.
 
thread_local! {
    static RNG: RefCell<SmallRng> = RefCell::new(SmallRng::seed_from_u64(system_seed()));
}
 
/// Produces a high-entropy seed using time, PID, and stack address mixing.
pub fn system_seed() -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
 
    let nanos = now.as_nanos() as u64;
    let pid = std::process::id() as u64;
    // Stack address adds ASLR entropy
    let addr = (&nanos as *const _) as u64;
 
    // SplitMix64 finalizer for avalanche effect
    let mut x = nanos ^ pid.rotate_left(17) ^ addr.rotate_right(11);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}
 
/// Reseed the calling thread's RNG with the given seed.
/// Results are reproducible on that thread for the same seed.
pub fn srand(seed: u64) {
    RNG.with(|rng| *rng.borrow_mut() = SmallRng::seed_from_u64(seed));
}
 
/// Random u64 in [0, u64::MAX].
pub fn rand_u64() -> u64 {
    RNG.with(|rng| rng.borrow_mut().gen::<u64>())
}
 
/// Random u64 in [min, max] (inclusive). Panics if min > max.
pub fn rand_range(min: u64, max: u64) -> u64 {
    assert!(min <= max, "rand_range: min ({min}) must be <= max ({max})");
    RNG.with(|rng| rng.borrow_mut().gen_range(min..=max))
}
 
/// Random bool with equal probability.
pub fn rand_bool() -> bool {
    RNG.with(|rng| rng.borrow_mut().gen::<bool>())
}
 
/// Random bool with a given probability of being true (0.0–1.0).
pub fn rand_bool_weighted(probability: f64) -> bool {
    debug_assert!((0.0..=1.0).contains(&probability), "probability must be in [0.0, 1.0]");
    rand_f64() < probability
}
 
/// Random f64 in [0.0, 1.0).
pub fn rand_f64() -> f64 {
    RNG.with(|rng| rng.borrow_mut().gen::<f64>())
}
 
/// Random f64 in [min, max). Panics if min > max.
pub fn rand_f64_range(min: f64, max: f64) -> f64 {
    assert!(min <= max, "rand_f64_range: min must be <= max");
    RNG.with(|rng| rng.borrow_mut().gen_range(min..max))
}
 
/// Shuffle a mutable slice in place (Fisher-Yates).
pub fn shuffle<T>(slice: &mut [T]) {
    let len = slice.len();
    for i in (1..len).rev() {
        let j = rand_range(0, i as u64) as usize;
        slice.swap(i, j);
    }
}
 
/// Pick a random element from a slice. Returns None if the slice is empty.
pub fn choose<T>(slice: &[T]) -> Option<&T> {
    if slice.is_empty() {
        return None;
    }
    let i = rand_range(0, slice.len() as u64 - 1) as usize;
    Some(&slice[i])
}
 
// ----------------------
// BASIC ARITHMETIC
// ----------------------
//
// The original variadic iterator API is kept where it makes sense
// (sum/product over collections), but simple two-operand forms are
// added for the common case so callers don't have to build a Vec.
 
/// Sum all values in an iterator.
pub fn sum<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    values.into_iter().sum()
}
 
/// Subtract each subsequent value from the first.
/// Returns 0.0 for an empty iterator.
pub fn subtract<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let mut iter = values.into_iter();
    let first = iter.next().unwrap_or(0.0);
    iter.fold(first, |acc, x| acc - x)
}
 
/// Multiply all values in an iterator.
pub fn product<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    values.into_iter().product()
}
 
/// Divide the first value by each subsequent value.
/// Returns 1.0 for an empty iterator.
pub fn divide<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let mut iter = values.into_iter();
    let first = iter.next().unwrap_or(1.0);
    iter.fold(first, |acc, x| acc / x)
}
 
// ----------------------
// POWERS & ROOTS
// ----------------------
 
pub fn power(base: f64, exponent: f64) -> f64  { base.powf(exponent) }
pub fn squared(value: f64) -> f64              { value * value }
pub fn cubed(value: f64) -> f64                { value * value * value }
pub fn sqrt(value: f64) -> f64                 { value.sqrt() }
pub fn cbrt(value: f64) -> f64                 { value.cbrt() }
pub fn hypot(x: f64, y: f64) -> f64            { x.hypot(y) }
 
// ----------------------
// LOGARITHMS & EXPONENTIALS
// ----------------------
 
pub fn exp(value: f64) -> f64                  { value.exp() }
pub fn exp2(value: f64) -> f64                 { value.exp2() }
pub fn ln(value: f64) -> f64                   { value.ln() }
pub fn log2(value: f64) -> f64                 { value.log2() }
pub fn log10(value: f64) -> f64                { value.log10() }
pub fn log(value: f64, base: f64) -> f64       { value.log(base) }
 
// ----------------------
// TRIGONOMETRY (radians)
// ----------------------
 
pub fn sin(value: f64) -> f64                  { value.sin() }
pub fn cos(value: f64) -> f64                  { value.cos() }
pub fn tan(value: f64) -> f64                  { value.tan() }
pub fn asin(value: f64) -> f64                 { value.asin() }
pub fn acos(value: f64) -> f64                 { value.acos() }
pub fn atan(value: f64) -> f64                 { value.atan() }
pub fn atan2(y: f64, x: f64) -> f64            { y.atan2(x) }
pub fn sinh(value: f64) -> f64                 { value.sinh() }
pub fn cosh(value: f64) -> f64                 { value.cosh() }
pub fn tanh(value: f64) -> f64                 { value.tanh() }
 
/// Convert degrees to radians.
pub fn to_radians(degrees: f64) -> f64         { degrees.to_radians() }
/// Convert radians to degrees.
pub fn to_degrees(radians: f64) -> f64         { radians.to_degrees() }
 
// ----------------------
// ROUNDING & CLAMPING
// ----------------------
 
pub fn abs(value: f64) -> f64                  { value.abs() }
pub fn floor(value: f64) -> f64                { value.floor() }
pub fn ceil(value: f64) -> f64                 { value.ceil() }
pub fn round(value: f64) -> f64                { value.round() }
pub fn trunc(value: f64) -> f64                { value.trunc() }
pub fn fract(value: f64) -> f64                { value.fract() }
pub fn signum(value: f64) -> f64               { value.signum() }
pub fn clamp(value: f64, min: f64, max: f64) -> f64 { value.clamp(min, max) }
 
// ----------------------
// INTERPOLATION
// ----------------------
 
/// Linear interpolation between `a` and `b` by factor `t` in [0.0, 1.0].
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}
 
/// Inverse lerp: given a value in [a, b], return the corresponding t in [0.0, 1.0].
pub fn inv_lerp(a: f64, b: f64, value: f64) -> f64 {
    if (b - a).abs() < f64::EPSILON {
        0.0
    } else {
        (value - a) / (b - a)
    }
}
 
/// Remap `value` from [in_min, in_max] to [out_min, out_max].
pub fn remap(value: f64, in_min: f64, in_max: f64, out_min: f64, out_max: f64) -> f64 {
    lerp(out_min, out_max, inv_lerp(in_min, in_max, value))
}
 
/// Smoothstep (Ken Perlin's version) — smooth Hermite interpolation.
pub fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = clamp(inv_lerp(edge0, edge1, x), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
 
// ----------------------
// COLLECTION HELPERS
// ----------------------
 
/// Maximum value in an iterator. Returns f64::NEG_INFINITY for empty input.
pub fn max<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    values.into_iter().fold(f64::NEG_INFINITY, f64::max)
}
 
/// Minimum value in an iterator. Returns f64::INFINITY for empty input.
pub fn min<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    values.into_iter().fold(f64::INFINITY, f64::min)
}
 
/// Range (max − min) of values in an iterator.
pub fn range<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let v: Vec<f64> = values.into_iter().collect();
    max(v.iter().copied()) - min(v.iter().copied())
}
 
// ----------------------
// STATISTICS
// ----------------------
 
/// Arithmetic mean. Returns 0.0 for empty input.
pub fn mean<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let mut count: usize = 0;
    let sum: f64 = values.into_iter().inspect(|_| count += 1).sum();
    if count == 0 { 0.0 } else { sum / count as f64 }
}
 
/// Alias for `mean`.
pub fn average<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    mean(values)
}
 
/// Median of a collection. Returns 0.0 for empty input.
pub fn median<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let mut v: Vec<f64> = values.into_iter().collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    if v.len() % 2 == 0 {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}
 
/// Mode(s) of a collection.
///
/// NaN values are grouped together (they compare equal by bit pattern here,
/// which is intentional for mode purposes — if you need NaN-aware logic,
/// filter them out before calling).
pub fn mode<I: IntoIterator<Item = f64>>(values: I) -> Vec<f64> {
    let mut counts: HashMap<u64, usize> = HashMap::new();
    for v in values {
        *counts.entry(v.to_bits()).or_insert(0) += 1;
    }
    let max_count = counts.values().copied().max().unwrap_or(0);
    let mut modes: Vec<f64> = counts
        .into_iter()
        .filter(|&(_, c)| c == max_count)
        .map(|(bits, _)| f64::from_bits(bits))
        .collect();
    modes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    modes
}
 
/// Population variance (divide by N). Returns 0.0 for empty input.
pub fn variance<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let v: Vec<f64> = values.into_iter().collect();
    if v.is_empty() {
        return 0.0;
    }
    let m = mean(v.iter().copied());
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64
}
 
/// Sample variance (divide by N−1, Bessel's correction). Returns 0.0 for fewer than 2 values.
pub fn sample_variance<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let v: Vec<f64> = values.into_iter().collect();
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v.iter().copied());
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64
}
 
/// Population standard deviation.
pub fn stddev<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    variance(values).sqrt()
}
 
/// Sample standard deviation (Bessel-corrected).
pub fn sample_stddev<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    sample_variance(values).sqrt()
}
 
/// Interquartile range (Q3 − Q1).
pub fn iqr<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let mut v: Vec<f64> = values.into_iter().collect();
    if v.len() < 4 {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q1 = percentile_sorted(&v, 25.0);
    let q3 = percentile_sorted(&v, 75.0);
    q3 - q1
}
 
/// Percentile of a collection (0–100). Uses linear interpolation.
pub fn percentile<I: IntoIterator<Item = f64>>(values: I, p: f64) -> f64 {
    let mut v: Vec<f64> = values.into_iter().collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile_sorted(&v, p)
}
 
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let p = p.clamp(0.0, 100.0);
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let rank = p / 100.0 * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    lerp(sorted[lo], sorted[hi], rank - lo as f64)
}
 
/// Pearson correlation coefficient between two equally-sized collections.
/// Returns None if either is empty or they differ in length.
pub fn correlation<I, J>(xs: I, ys: J) -> Option<f64>
where
    I: IntoIterator<Item = f64>,
    J: IntoIterator<Item = f64>,
{
    let xs: Vec<f64> = xs.into_iter().collect();
    let ys: Vec<f64> = ys.into_iter().collect();
    if xs.is_empty() || xs.len() != ys.len() {
        return None;
    }
    let mx = mean(xs.iter().copied());
    let my = mean(ys.iter().copied());
    let num: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| (x - mx) * (y - my)).sum();
    let den = (xs.iter().map(|x| (x - mx).powi(2)).sum::<f64>()
        * ys.iter().map(|y| (y - my).powi(2)).sum::<f64>())
    .sqrt();
    if den == 0.0 { None } else { Some(num / den) }
}
 
/// Covariance between two equally-sized collections (population).
/// Returns None if they differ in length or are empty.
pub fn covariance<I, J>(xs: I, ys: J) -> Option<f64>
where
    I: IntoIterator<Item = f64>,
    J: IntoIterator<Item = f64>,
{
    let xs: Vec<f64> = xs.into_iter().collect();
    let ys: Vec<f64> = ys.into_iter().collect();
    if xs.is_empty() || xs.len() != ys.len() {
        return None;
    }
    let mx = mean(xs.iter().copied());
    let my = mean(ys.iter().copied());
    let cov: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| (x - mx) * (y - my)).sum();
    Some(cov / xs.len() as f64)
}
 
// ----------------------
// NUMBER THEORY
// ----------------------
 
/// Greatest common divisor (Euclidean algorithm).
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
 
/// Least common multiple.
pub fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        return 0;
    }
    a / gcd(a, b) * b
}
 
/// Check if a number is prime (trial division — fine for small n).
pub fn is_prime(n: u64) -> bool {
    if n < 2 { return false; }
    if n < 4 { return true; }
    if n % 2 == 0 || n % 3 == 0 { return false; }
    let mut i = 5u64;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 { return false; }
        i += 6;
    }
    true
}
 
/// Factorial of n. Panics on overflow for large n (use with care).
pub fn factorial(n: u64) -> u64 {
    (1..=n).product()
}
 
/// Number of combinations C(n, k).
pub fn combinations(n: u64, k: u64) -> u64 {
    if k > n { return 0; }
    let k = k.min(n - k); // symmetry
    (0..k).fold(1u64, |acc, i| acc * (n - i) / (i + 1))
}
 
/// Number of permutations P(n, k).
pub fn permutations(n: u64, k: u64) -> u64 {
    if k > n { return 0; }
    (n - k + 1..=n).product()
}
 
// ----------------------
// CONSTANTS
// ----------------------
 
pub mod consts {
    pub const PI: f64      = std::f64::consts::PI;
    pub const TAU: f64     = std::f64::consts::TAU;
    pub const E: f64       = std::f64::consts::E;
    pub const SQRT_2: f64  = std::f64::consts::SQRT_2;
    pub const LN_2: f64    = std::f64::consts::LN_2;
    pub const LN_10: f64   = std::f64::consts::LN_10;
    pub const PHI: f64     = 1.618_033_988_749_895; // golden ratio
}
 
// ----------------------
// TESTS
// ----------------------
 
#[cfg(test)]
mod tests {
    use super::*;
 
    #[test]
    fn test_basic_arithmetic() {
        assert_eq!(sum([1.0, 2.0, 3.0]), 6.0);
        assert_eq!(subtract([10.0, 3.0, 2.0]), 5.0);
        assert_eq!(product([2.0, 3.0, 4.0]), 24.0);
        assert_eq!(divide([100.0, 5.0, 4.0]), 5.0);
    }
 
    #[test]
    fn test_statistics() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean(data.iter().copied()), 3.0);
        assert_eq!(median(data.iter().copied()), 3.0);
        assert_eq!(variance(data.iter().copied()), 2.0);
        assert!((stddev(data.iter().copied()) - std::f64::consts::SQRT_2).abs() < 1e-10);
    }
 
    #[test]
    fn test_sample_variance() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((sample_variance(data.iter().copied()) - 4.571_428).abs() < 1e-5);
    }
 
    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(data.iter().copied(), 0.0), 1.0);
        assert_eq!(percentile(data.iter().copied(), 100.0), 10.0);
        assert_eq!(percentile(data.iter().copied(), 50.0), 5.5);
    }
 
    #[test]
    fn test_correlation() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![2.0, 4.0, 6.0];
        assert!((correlation(xs, ys).unwrap() - 1.0).abs() < 1e-10);
    }
 
    #[test]
    fn test_gcd_lcm() {
        assert_eq!(gcd(48, 18), 6);
        assert_eq!(lcm(4, 6), 12);
    }
 
    #[test]
    fn test_prime() {
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(97));
        assert!(!is_prime(100));
    }
 
    #[test]
    fn test_combinations_permutations() {
        assert_eq!(combinations(5, 2), 10);
        assert_eq!(permutations(5, 2), 20);
    }
 
    #[test]
    fn test_lerp_remap() {
        assert_eq!(lerp(0.0, 10.0, 0.5), 5.0);
        assert_eq!(remap(5.0, 0.0, 10.0, 0.0, 100.0), 50.0);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < 1e-10);
    }
 
    #[test]
    fn test_rand_reproducible() {
        srand(42);
        let a = rand_u64();
        srand(42);
        let b = rand_u64();
        assert_eq!(a, b, "srand should make results reproducible");
    }
 
    #[test]
    fn test_rand_range() {
        for _ in 0..1000 {
            let v = rand_range(5, 10);
            assert!((5..=10).contains(&v));
        }
    }
 
    #[test]
    fn test_shuffle() {
        let mut v = vec![1, 2, 3, 4, 5];
        let original = v.clone();
        shuffle(&mut v);
        // Same elements, possibly different order
        let mut sorted = v.clone();
        sorted.sort();
        assert_eq!(sorted, original);
    }
 
    #[test]
    fn test_trig() {
        assert!((sin(consts::PI) - 0.0).abs() < 1e-10);
        assert!((cos(0.0) - 1.0).abs() < 1e-10);
        assert!((to_degrees(consts::PI) - 180.0).abs() < 1e-10);
    }
}

// ============================================================
//  UNIVERSAL VECTOR GENERATOR (Vec2, Vec3, Vec4)
// ============================================================

use std::ops::{
    Add, AddAssign,
    Sub, SubAssign,
    Mul, MulAssign,
    Div, DivAssign,
    Neg,
};

macro_rules! define_vec {
    ($name:ident, $($field:ident),+) => {

        #[repr(C)]
        #[derive(Debug, Clone, Copy, PartialEq, Default)]
        pub struct $name {
            $(pub $field: f64,)+
        }

        impl $name {

            // ------------------------------------------------
            // Constructors
            // ------------------------------------------------

            pub fn new($($field: f64),+) -> Self {
                Self { $($field),+ }
            }

            pub fn zero() -> Self {
                Self {
                    $($field: 0.0),+
                }
            }

            pub fn one() -> Self {
                Self {
                    $($field: 1.0),+
                }
            }

            // ------------------------------------------------
            // Magnitude
            // ------------------------------------------------

            pub fn length(self) -> f64 {
                self.length_squared().sqrt()
            }

            pub fn length_squared(self) -> f64 {
                let mut s = 0.0;

                $(
                    s += self.$field * self.$field;
                )+

                s
            }

            pub fn normalized(self) -> Self {
                let len = self.length();

                if len == 0.0 {
                    Self::zero()
                } else {
                    self / len
                }
            }

            // ------------------------------------------------
            // Math
            // ------------------------------------------------

            pub fn dot(self, other: Self) -> f64 {
                let mut s = 0.0;

                $(
                    s += self.$field * other.$field;
                )+

                s
            }

            pub fn distance(self, other: Self) -> f64 {
                (self - other).length()
            }

            pub fn distance_squared(self, other: Self) -> f64 {
                (self - other).length_squared()
            }

            // ------------------------------------------------
            // Interpolation
            // ------------------------------------------------

            pub fn lerp(self, to: Self, t: f64) -> Self {
                Self {
                    $(
                        $field: self.$field + (to.$field - self.$field) * t
                    ),+
                }
            }

            pub fn smoothstep(self, to: Self, t: f64) -> Self {
                let t = t * t * (3.0 - 2.0 * t);
                self.lerp(to, t)
            }

            // ------------------------------------------------
            // Projection / Reflection
            // ------------------------------------------------

            pub fn project(self, onto: Self) -> Self {
                let denom = onto.length_squared();

                if denom == 0.0 {
                    Self::zero()
                } else {
                    onto * (self.dot(onto) / denom)
                }
            }

            // normal should be normalized
            pub fn reflect(self, normal: Self) -> Self {
                self - normal * (2.0 * self.dot(normal))
            }

            // ------------------------------------------------
            // Clamp magnitude
            // ------------------------------------------------

            pub fn clamp_length(self, min: f64, max: f64) -> Self {

                let len = self.length();

                if len == 0.0 {
                    return Self::zero();
                }

                let norm = self / len;

                if len < min {
                    norm * min
                } else if len > max {
                    norm * max
                } else {
                    self
                }
            }

            // ------------------------------------------------
            // Component min/max
            // ------------------------------------------------

            pub fn min(self, other: Self) -> Self {
                Self {
                    $(
                        $field: self.$field.min(other.$field)
                    ),+
                }
            }

            pub fn max(self, other: Self) -> Self {
                Self {
                    $(
                        $field: self.$field.max(other.$field)
                    ),+
                }
            }

            // ------------------------------------------------
            // Random vectors
            // ------------------------------------------------

            pub fn random_range(min: f64, max: f64) -> Self {
                Self {
                    $(
                        $field: crate::rand_f64_range(min, max)
                    ),+
                }
            }

            pub fn random_unit() -> Self {

                loop {

                    let v = Self {
                        $(
                            $field: crate::rand_f64_range(-1.0, 1.0)
                        ),+
                    };

                    let len = v.length();

                    if len > 0.0 {
                        return v / len;
                    }
                }
            }
        }

        // ====================================================
        // Operators
        // ====================================================

        impl Add for $name {
            type Output = Self;

            fn add(self, rhs: Self) -> Self {
                Self {
                    $(
                        $field: self.$field + rhs.$field
                    ),+
                }
            }
        }

        impl AddAssign for $name {

            fn add_assign(&mut self, rhs: Self) {
                $(
                    self.$field += rhs.$field;
                )+
            }
        }

        impl Sub for $name {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self {
                Self {
                    $(
                        $field: self.$field - rhs.$field
                    ),+
                }
            }
        }

        impl SubAssign for $name {

            fn sub_assign(&mut self, rhs: Self) {
                $(
                    self.$field -= rhs.$field;
                )+
            }
        }

        impl Mul<f64> for $name {
            type Output = Self;

            fn mul(self, rhs: f64) -> Self {
                Self {
                    $(
                        $field: self.$field * rhs
                    ),+
                }
            }
        }

        impl MulAssign<f64> for $name {

            fn mul_assign(&mut self, rhs: f64) {
                $(
                    self.$field *= rhs;
                )+
            }
        }

        impl Div<f64> for $name {
            type Output = Self;

            fn div(self, rhs: f64) -> Self {
                Self {
                    $(
                        $field: self.$field / rhs
                    ),+
                }
            }
        }

        impl DivAssign<f64> for $name {

            fn div_assign(&mut self, rhs: f64) {
                $(
                    self.$field /= rhs;
                )+
            }
        }

        impl Neg for $name {
            type Output = Self;

            fn neg(self) -> Self {
                Self {
                    $(
                        $field: -self.$field
                    ),+
                }
            }
        }
    };
}

// ============================================================
//  Generate vectors
// ============================================================

define_vec!(Vec2, x, y);
define_vec!(Vec3, x, y, z);
define_vec!(Vec4, x, y, z, w);

// ============================================================
//  Scalar * Vector
// ============================================================

impl Mul<Vec2> for f64 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Vec2 {
        rhs * self
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, rhs: Vec3) -> Vec3 {
        rhs * self
    }
}

impl Mul<Vec4> for f64 {
    type Output = Vec4;

    fn mul(self, rhs: Vec4) -> Vec4 {
        rhs * self
    }
}

// ============================================================
//  Vec2
// ============================================================

impl Vec2 {

    // -----------------------------------------
    // Swizzling
    // -----------------------------------------

    pub fn xy(self) -> Vec2 {
        self
    }

    // -----------------------------------------
    // Rotation
    // -----------------------------------------

    pub fn rotated(self, angle: f64) -> Self {

        let (s, c) = angle.sin_cos();

        Self {
            x: self.x * c - self.y * s,
            y: self.x * s + self.y * c,
        }
    }

    pub fn from_angle(angle: f64) -> Self {

        let (s, c) = angle.sin_cos();

        Self {
            x: c,
            y: s,
        }
    }

    pub fn perpendicular(self) -> Self {
        Self {
            x: -self.y,
            y: self.x,
        }
    }

    pub fn angle_to(self, other: Self) -> f64 {

        let dot = self.dot(other);
        let len = self.length() * other.length();

        if len == 0.0 {
            0.0
        } else {
            (dot / len).acos()
        }
    }
}

// ============================================================
//  Vec3
// ============================================================

impl Vec3 {

    // -----------------------------------------
    // Swizzling
    // -----------------------------------------

    pub fn xy(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    pub fn yz(self) -> Vec2 {
        Vec2::new(self.y, self.z)
    }

    pub fn xz(self) -> Vec2 {
        Vec2::new(self.x, self.z)
    }

    pub fn xyz(self) -> Vec3 {
        self
    }

    // -----------------------------------------
    // Cross product
    // -----------------------------------------

    pub fn cross(self, other: Self) -> Self {

        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn angle_to(self, other: Self) -> f64 {

        let dot = self.dot(other);
        let len = self.length() * other.length();

        if len == 0.0 {
            0.0
        } else {
            (dot / len).acos()
        }
    }
}

// ============================================================
//  Vec4
// ============================================================

impl Vec4 {

    // -----------------------------------------
    // Swizzling
    // -----------------------------------------

    pub fn xy(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    pub fn xyz(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

// -----------------------------------------
// Basic helpers
// -----------------------------------------

pub fn quat_identity() -> Quat {
    Quat::new(0.0, 0.0, 0.0, 1.0)
}

pub fn quat_normalize(q: Quat) -> Quat {
    let len = (q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w).sqrt();
    if len == 0.0 {
        quat_identity()
    } else {
        Quat::new(q.x/len, q.y/len, q.z/len, q.w/len)
    }
}

pub fn quat_conjugate(q: Quat) -> Quat {
    Quat::new(-q.x, -q.y, -q.z, q.w)
}

pub fn quat_inverse(q: Quat) -> Quat {
    let len_sq = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
    if len_sq == 0.0 {
        quat_identity()
    } else {
        let c = quat_conjugate(q);
        Quat::new(c.x/len_sq, c.y/len_sq, c.z/len_sq, c.w/len_sq)
    }
}

// -----------------------------------------
// From axis/angle
// -----------------------------------------

pub fn quat_from_axis_angle(angle: f64, axis: Vec3) -> Quat {
    let len = axis.length();
    if len == 0.0 {
        return quat_identity();
    }

    let axis = axis / len;
    let half = angle * 0.5;
    let (s, c) = half.sin_cos();

    Quat::new(axis.x * s, axis.y * s, axis.z * s, c)
}

// -----------------------------------------
// Quaternion multiplication (combine rotations)
// -----------------------------------------

pub fn quat_mul(a: Quat, b: Quat) -> Quat {
    Quat::new(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
    )
}

// -----------------------------------------
// Rotate Vec3 by quaternion
// -----------------------------------------

pub fn quat_rotate_vec3(q: Quat, v: Vec3) -> Vec3 {
    // v' = q * (v,0) * q^-1
    let qv = Vec3::new(q.x, q.y, q.z);
    let t = 2.0 * qv.cross(v);
    v + t * q.w + qv.cross(t)
}

// -----------------------------------------
// From Euler (yaw, pitch, roll)
// yaw = rotation around Y
// pitch = rotation around X
// roll = rotation around Z
// -----------------------------------------

//  MATRIX MODULE (Mat2, Mat3, Mat4)
// ============================================================
//
// Design goals:
//  - Match the style of Vec2 / Vec3 / Vec4
//  - Use f64 everywhere
//  - Row-major storage (m11, m12, ...)
//  - Provide common constructors (identity, zero, from_rows, from_cols)
//  - Provide basic operations (add, sub, mul, scalar mul/div)
//  - Provide determinant, transpose, inverse where applicable
//  - Provide vector transform helpers
//
//  This module assumes Vec2, Vec3, Vec4 already exist in the crate.
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat2 {
    pub m11: f64, pub m12: f64,
    pub m21: f64, pub m22: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub m11: f64, pub m12: f64, pub m13: f64,
    pub m21: f64, pub m22: f64, pub m23: f64,
    pub m31: f64, pub m32: f64, pub m33: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub m11: f64, pub m12: f64, pub m13: f64, pub m14: f64,
    pub m21: f64, pub m22: f64, pub m23: f64, pub m24: f64,
    pub m31: f64, pub m32: f64, pub m33: f64, pub m34: f64,
    pub m41: f64, pub m42: f64, pub m43: f64, pub m44: f64,
}

// ============================================================
//  Mat2
// ============================================================

impl Mat2 {
    pub fn new(
        m11: f64, m12: f64,
        m21: f64, m22: f64,
    ) -> Self {
        Self { m11, m12, m21, m22 }
    }

    pub fn zero() -> Self {
        Self {
            m11: 0.0, m12: 0.0,
            m21: 0.0, m22: 0.0,
        }
    }

    pub fn identity() -> Self {
        Self {
            m11: 1.0, m12: 0.0,
            m21: 0.0, m22: 1.0,
        }
    }

    pub fn from_rows(r1: (f64, f64), r2: (f64, f64)) -> Self {
        Self {
            m11: r1.0, m12: r1.1,
            m21: r2.0, m22: r2.1,
        }
    }

    pub fn from_cols(c1: (f64, f64), c2: (f64, f64)) -> Self {
        Self {
            m11: c1.0, m12: c2.0,
            m21: c1.1, m22: c2.1,
        }
    }

    pub fn row(&self, i: usize) -> (f64, f64) {
        match i {
            0 => (self.m11, self.m12),
            1 => (self.m21, self.m22),
            _ => panic!("Mat2::row index out of range"),
        }
    }

    pub fn col(&self, i: usize) -> (f64, f64) {
        match i {
            0 => (self.m11, self.m21),
            1 => (self.m12, self.m22),
            _ => panic!("Mat2::col index out of range"),
        }
    }

    pub fn transpose(&self) -> Self {
        Self {
            m11: self.m11, m12: self.m21,
            m21: self.m12, m22: self.m22,
        }
    }

    pub fn determinant(&self) -> f64 {
        self.m11 * self.m22 - self.m12 * self.m21
    }

    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < f64::EPSILON {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Self {
            m11:  self.m22 * inv_det,
            m12: -self.m12 * inv_det,
            m21: -self.m21 * inv_det,
            m22:  self.m11 * inv_det,
        })
    }

    pub fn mul_vec2(&self, v: crate::Vec2) -> crate::Vec2 {
        crate::Vec2::new(
            self.m11 * v.x + self.m12 * v.y,
            self.m21 * v.x + self.m22 * v.y,
        )
    }

    pub fn scale(sx: f64, sy: f64) -> Self {
        Self {
            m11: sx,  m12: 0.0,
            m21: 0.0, m22: sy,
        }
    }

    pub fn rotation(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            m11: c,  m12: -s,
            m21: s,  m22:  c,
        }
    }
}

// ============================================================
//  Mat3
// ============================================================

impl Mat3 {
    pub fn new(
        m11: f64, m12: f64, m13: f64,
        m21: f64, m22: f64, m23: f64,
        m31: f64, m32: f64, m33: f64,
    ) -> Self {
        Self { m11, m12, m13, m21, m22, m23, m31, m32, m33 }
    }

    pub fn zero() -> Self {
        Self {
            m11: 0.0, m12: 0.0, m13: 0.0,
            m21: 0.0, m22: 0.0, m23: 0.0,
            m31: 0.0, m32: 0.0, m33: 0.0,
        }
    }

    pub fn identity() -> Self {
        Self {
            m11: 1.0, m12: 0.0, m13: 0.0,
            m21: 0.0, m22: 1.0, m23: 0.0,
            m31: 0.0, m32: 0.0, m33: 1.0,
        }
    }

    pub fn from_rows(
        r1: (f64, f64, f64),
        r2: (f64, f64, f64),
        r3: (f64, f64, f64),
    ) -> Self {
        Self {
            m11: r1.0, m12: r1.1, m13: r1.2,
            m21: r2.0, m22: r2.1, m23: r2.2,
            m31: r3.0, m32: r3.1, m33: r3.2,
        }
    }

    pub fn from_cols(
        c1: (f64, f64, f64),
        c2: (f64, f64, f64),
        c3: (f64, f64, f64),
    ) -> Self {
        Self {
            m11: c1.0, m12: c2.0, m13: c3.0,
            m21: c1.1, m22: c2.1, m23: c3.1,
            m31: c1.2, m32: c2.2, m33: c3.2,
        }
    }

    pub fn row(&self, i: usize) -> (f64, f64, f64) {
        match i {
            0 => (self.m11, self.m12, self.m13),
            1 => (self.m21, self.m22, self.m23),
            2 => (self.m31, self.m32, self.m33),
            _ => panic!("Mat3::row index out of range"),
        }
    }

    pub fn col(&self, i: usize) -> (f64, f64, f64) {
        match i {
            0 => (self.m11, self.m21, self.m31),
            1 => (self.m12, self.m22, self.m32),
            2 => (self.m13, self.m23, self.m33),
            _ => panic!("Mat3::col index out of range"),
        }
    }

    pub fn transpose(&self) -> Self {
        Self {
            m11: self.m11, m12: self.m21, m13: self.m31,
            m21: self.m12, m22: self.m22, m23: self.m32,
            m31: self.m13, m32: self.m23, m33: self.m33,
        }
    }

    pub fn determinant(&self) -> f64 {
        self.m11 * (self.m22 * self.m33 - self.m23 * self.m32)
      - self.m12 * (self.m21 * self.m33 - self.m23 * self.m31)
      + self.m13 * (self.m21 * self.m32 - self.m22 * self.m31)
    }

    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < f64::EPSILON {
            return None;
        }
        let inv_det = 1.0 / det;

        let m11 =  (self.m22 * self.m33 - self.m23 * self.m32) * inv_det;
        let m12 = -(self.m12 * self.m33 - self.m13 * self.m32) * inv_det;
        let m13 =  (self.m12 * self.m23 - self.m13 * self.m22) * inv_det;

        let m21 = -(self.m21 * self.m33 - self.m23 * self.m31) * inv_det;
        let m22 =  (self.m11 * self.m33 - self.m13 * self.m31) * inv_det;
        let m23 = -(self.m11 * self.m23 - self.m13 * self.m21) * inv_det;

        let m31 =  (self.m21 * self.m32 - self.m22 * self.m31) * inv_det;
        let m32 = -(self.m11 * self.m32 - self.m12 * self.m31) * inv_det;
        let m33 =  (self.m11 * self.m22 - self.m12 * self.m21) * inv_det;

        Some(Self {
            m11, m12, m13,
            m21, m22, m23,
            m31, m32, m33,
        })
    }

    pub fn mul_vec3(&self, v: crate::Vec3) -> crate::Vec3 {
        crate::Vec3::new(
            self.m11 * v.x + self.m12 * v.y + self.m13 * v.z,
            self.m21 * v.x + self.m22 * v.y + self.m23 * v.z,
            self.m31 * v.x + self.m32 * v.y + self.m33 * v.z,
        )
    }

    pub fn scale(sx: f64, sy: f64, sz: f64) -> Self {
        Self {
            m11: sx,  m12: 0.0, m13: 0.0,
            m21: 0.0, m22: sy,  m23: 0.0,
            m31: 0.0, m32: 0.0, m33: sz,
        }
    }

    pub fn rotation_x(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            m11: 1.0, m12: 0.0, m13: 0.0,
            m21: 0.0, m22: c,   m23: -s,
            m31: 0.0, m32: s,   m33:  c,
        }
    }

    pub fn rotation_y(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            m11:  c,  m12: 0.0, m13: s,
            m21: 0.0, m22: 1.0, m23: 0.0,
            m31: -s,  m32: 0.0, m33: c,
        }
    }

    pub fn rotation_z(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            m11: c,   m12: -s,  m13: 0.0,
            m21: s,   m22:  c,  m23: 0.0,
            m31: 0.0, m32: 0.0, m33: 1.0,
        }
    }
}

// ============================================================
//  Mat4
// ============================================================

impl Mat4 {
    pub fn new(
        m11: f64, m12: f64, m13: f64, m14: f64,
        m21: f64, m22: f64, m23: f64, m24: f64,
        m31: f64, m32: f64, m33: f64, m34: f64,
        m41: f64, m42: f64, m43: f64, m44: f64,
    ) -> Self {
        Self {
            m11, m12, m13, m14,
            m21, m22, m23, m24,
            m31, m32, m33, m34,
            m41, m42, m43, m44,
        }
    }

    pub fn zero() -> Self {
        Self {
            m11: 0.0, m12: 0.0, m13: 0.0, m14: 0.0,
            m21: 0.0, m22: 0.0, m23: 0.0, m24: 0.0,
            m31: 0.0, m32: 0.0, m33: 0.0, m34: 0.0,
            m41: 0.0, m42: 0.0, m43: 0.0, m44: 0.0,
        }
    }

    pub fn identity() -> Self {
        Self {
            m11: 1.0, m12: 0.0, m13: 0.0, m14: 0.0,
            m21: 0.0, m22: 1.0, m23: 0.0, m24: 0.0,
            m31: 0.0, m32: 0.0, m33: 1.0, m34: 0.0,
            m41: 0.0, m42: 0.0, m43: 0.0, m44: 1.0,
        }
    }

    pub fn from_rows(
        r1: (f64, f64, f64, f64),
        r2: (f64, f64, f64, f64),
        r3: (f64, f64, f64, f64),
        r4: (f64, f64, f64, f64),
    ) -> Self {
        Self {
            m11: r1.0, m12: r1.1, m13: r1.2, m14: r1.3,
            m21: r2.0, m22: r2.1, m23: r2.2, m24: r2.3,
            m31: r3.0, m32: r3.1, m33: r3.2, m34: r3.3,
            m41: r4.0, m42: r4.1, m43: r4.2, m44: r4.3,
        }
    }

    pub fn from_cols(
        c1: (f64, f64, f64, f64),
        c2: (f64, f64, f64, f64),
        c3: (f64, f64, f64, f64),
        c4: (f64, f64, f64, f64),
    ) -> Self {
        Self {
            m11: c1.0, m12: c2.0, m13: c3.0, m14: c4.0,
            m21: c1.1, m22: c2.1, m23: c3.1, m24: c4.1,
            m31: c1.2, m32: c2.2, m33: c3.2, m34: c4.2,
            m41: c1.3, m42: c2.3, m43: c3.3, m44: c4.3,
        }
    }

    pub fn row(&self, i: usize) -> (f64, f64, f64, f64) {
        match i {
            0 => (self.m11, self.m12, self.m13, self.m14),
            1 => (self.m21, self.m22, self.m23, self.m24),
            2 => (self.m31, self.m32, self.m33, self.m34),
            3 => (self.m41, self.m42, self.m43, self.m44),
            _ => panic!("Mat4::row index out of range"),
        }
    }

    pub fn col(&self, i: usize) -> (f64, f64, f64, f64) {
        match i {
            0 => (self.m11, self.m21, self.m31, self.m41),
            1 => (self.m12, self.m22, self.m32, self.m42),
            2 => (self.m13, self.m23, self.m33, self.m43),
            3 => (self.m14, self.m24, self.m34, self.m44),
            _ => panic!("Mat4::col index out of range"),
        }
    }

    pub fn transpose(&self) -> Self {
        Self {
            m11: self.m11, m12: self.m21, m13: self.m31, m14: self.m41,
            m21: self.m12, m22: self.m22, m23: self.m32, m24: self.m42,
            m31: self.m13, m32: self.m23, m33: self.m33, m34: self.m43,
            m41: self.m14, m42: self.m24, m43: self.m34, m44: self.m44,
        }
    }

    pub fn mul_vec4(&self, v: crate::Vec4) -> crate::Vec4 {
        crate::Vec4::new(
            self.m11 * v.x + self.m12 * v.y + self.m13 * v.z + self.m14 * v.w,
            self.m21 * v.x + self.m22 * v.y + self.m23 * v.z + self.m24 * v.w,
            self.m31 * v.x + self.m32 * v.y + self.m33 * v.z + self.m34 * v.w,
            self.m41 * v.x + self.m42 * v.y + self.m43 * v.z + self.m44 * v.w,
        )
    }

    pub fn mul_vec3_point(&self, v: crate::Vec3) -> crate::Vec3 {
        let x = self.m11 * v.x + self.m12 * v.y + self.m13 * v.z + self.m14;
        let y = self.m21 * v.x + self.m22 * v.y + self.m23 * v.z + self.m24;
        let z = self.m31 * v.x + self.m32 * v.y + self.m33 * v.z + self.m34;
        let w = self.m41 * v.x + self.m42 * v.y + self.m43 * v.z + self.m44;

        if w.abs() < f64::EPSILON {
            crate::Vec3::new(x, y, z)
        } else {
            crate::Vec3::new(x / w, y / w, z / w)
        }
    }

    pub fn mul_vec3_direction(&self, v: crate::Vec3) -> crate::Vec3 {
        crate::Vec3::new(
            self.m11 * v.x + self.m12 * v.y + self.m13 * v.z,
            self.m21 * v.x + self.m22 * v.y + self.m23 * v.z,
            self.m31 * v.x + self.m32 * v.y + self.m33 * v.z,
        )
    }

    pub fn translation(tx: f64, ty: f64, tz: f64) -> Self {
        Self {
            m11: 1.0, m12: 0.0, m13: 0.0, m14: tx,
            m21: 0.0, m22: 1.0, m23: 0.0, m24: ty,
            m31: 0.0, m32: 0.0, m33: 1.0, m34: tz,
            m41: 0.0, m42: 0.0, m43: 0.0, m44: 1.0,
        }
    }

    pub fn scale(sx: f64, sy: f64, sz: f64) -> Self {
        Self {
            m11: sx,  m12: 0.0, m13: 0.0, m14: 0.0,
            m21: 0.0, m22: sy,  m23: 0.0, m24: 0.0,
            m31: 0.0, m32: 0.0, m33: sz,  m34: 0.0,
            m41: 0.0, m42: 0.0, m43: 0.0, m44: 1.0,
        }
    }

    pub fn rotation_x(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            m11: 1.0, m12: 0.0, m13: 0.0, m14: 0.0,
            m21: 0.0, m22: c,   m23: -s,  m24: 0.0,
            m31: 0.0, m32: s,   m33:  c,  m34: 0.0,
            m41: 0.0, m42: 0.0, m43: 0.0, m44: 1.0,
        }
    }

    pub fn rotation_y(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            m11:  c,  m12: 0.0, m13: s,   m14: 0.0,
            m21: 0.0, m22: 1.0, m23: 0.0, m24: 0.0,
            m31: -s,  m32: 0.0, m33: c,   m34: 0.0,
            m41: 0.0, m42: 0.0, m43: 0.0, m44: 1.0,
        }
    }

    pub fn rotation_z(angle: f64) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            m11: c,   m12: -s,  m13: 0.0, m14: 0.0,
            m21: s,   m22:  c,  m23: 0.0, m24: 0.0,
            m31: 0.0, m32: 0.0, m33: 1.0, m34: 0.0,
            m41: 0.0, m42: 0.0, m43: 0.0, m44: 1.0,
        }
    }

    pub fn perspective(fov_y_radians: f64, aspect: f64, near: f64, far: f64) -> Self {
        let f = 1.0 / (fov_y_radians / 2.0).tan();
        let nf = 1.0 / (near - far);

        Self {
            m11: f / aspect, m12: 0.0, m13: 0.0,                     m14: 0.0,
            m21: 0.0,        m22: f,   m23: 0.0,                     m24: 0.0,
            m31: 0.0,        m32: 0.0, m33: (far + near) * nf,       m34: 2.0 * far * near * nf,
            m41: 0.0,        m42: 0.0, m43: -1.0,                    m44: 0.0,
        }
    }

    pub fn orthographic(
        left: f64, right: f64,
        bottom: f64, top: f64,
        near: f64, far: f64,
    ) -> Self {
        let rl = right - left;
        let tb = top - bottom;
        let fn_ = far - near;

        Self {
            m11: 2.0 / rl, m12: 0.0,       m13: 0.0,        m14: -(right + left) / rl,
            m21: 0.0,      m22: 2.0 / tb,  m23: 0.0,        m24: -(top + bottom) / tb,
            m31: 0.0,      m32: 0.0,       m33: -2.0 / fn_, m34: -(far + near) / fn_,
            m41: 0.0,      m42: 0.0,       m43: 0.0,        m44: 1.0,
        }
    }
}

// ============================================================
//  Operators
// ============================================================

impl Add for Mat2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            m11: self.m11 + rhs.m11, m12: self.m12 + rhs.m12,
            m21: self.m21 + rhs.m21, m22: self.m22 + rhs.m22,
        }
    }
}

impl AddAssign for Mat2 {
    fn add_assign(&mut self, rhs: Self) {
        self.m11 += rhs.m11; self.m12 += rhs.m12;
        self.m21 += rhs.m21; self.m22 += rhs.m22;
    }
}

impl Sub for Mat2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            m11: self.m11 - rhs.m11, m12: self.m12 - rhs.m12,
            m21: self.m21 - rhs.m21, m22: self.m22 - rhs.m22,
        }
    }
}

impl SubAssign for Mat2 {
    fn sub_assign(&mut self, rhs: Self) {
        self.m11 -= rhs.m11; self.m12 -= rhs.m12;
        self.m21 -= rhs.m21; self.m22 -= rhs.m22;
    }
}

impl Mul<f64> for Mat2 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            m11: self.m11 * rhs, m12: self.m12 * rhs,
            m21: self.m21 * rhs, m22: self.m22 * rhs,
        }
    }
}

impl MulAssign<f64> for Mat2 {
    fn mul_assign(&mut self, rhs: f64) {
        self.m11 *= rhs; self.m12 *= rhs;
        self.m21 *= rhs; self.m22 *= rhs;
    }
}

impl Mul for Mat2 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            m11: self.m11 * rhs.m11 + self.m12 * rhs.m21,
            m12: self.m11 * rhs.m12 + self.m12 * rhs.m22,
            m21: self.m21 * rhs.m11 + self.m22 * rhs.m21,
            m22: self.m21 * rhs.m12 + self.m22 * rhs.m22,
        }
    }
}

// ------------------------------------------------------------

impl Add for Mat3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            m11: self.m11 + rhs.m11, m12: self.m12 + rhs.m12, m13: self.m13 + rhs.m13,
            m21: self.m21 + rhs.m21, m22: self.m22 + rhs.m22, m23: self.m23 + rhs.m23,
            m31: self.m31 + rhs.m31, m32: self.m32 + rhs.m32, m33: self.m33 + rhs.m33,
        }
    }
}

impl AddAssign for Mat3 {
    fn add_assign(&mut self, rhs: Self) {
        self.m11 += rhs.m11; self.m12 += rhs.m12; self.m13 += rhs.m13;
        self.m21 += rhs.m21; self.m22 += rhs.m22; self.m23 += rhs.m23;
        self.m31 += rhs.m31; self.m32 += rhs.m32; self.m33 += rhs.m33;
    }
}

impl Sub for Mat3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            m11: self.m11 - rhs.m11, m12: self.m12 - rhs.m12, m13: self.m13 - rhs.m13,
            m21: self.m21 - rhs.m21, m22: self.m22 - rhs.m22, m23: self.m23 - rhs.m23,
            m31: self.m31 - rhs.m31, m32: self.m32 - rhs.m32, m33: self.m33 - rhs.m33,
        }
    }
}

impl SubAssign for Mat3 {
    fn sub_assign(&mut self, rhs: Self) {
        self.m11 -= rhs.m11; self.m12 -= rhs.m12; self.m13 -= rhs.m13;
        self.m21 -= rhs.m21; self.m22 -= rhs.m22; self.m23 -= rhs.m23;
        self.m31 -= rhs.m31; self.m32 -= rhs.m32; self.m33 -= rhs.m33;
    }
}

impl Mul<f64> for Mat3 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            m11: self.m11 * rhs, m12: self.m12 * rhs, m13: self.m13 * rhs,
            m21: self.m21 * rhs, m22: self.m22 * rhs, m23: self.m23 * rhs,
            m31: self.m31 * rhs, m32: self.m32 * rhs, m33: self.m33 * rhs,
        }
    }
}

impl MulAssign<f64> for Mat3 {
    fn mul_assign(&mut self, rhs: f64) {
        self.m11 *= rhs; self.m12 *= rhs; self.m13 *= rhs;
        self.m21 *= rhs; self.m22 *= rhs; self.m23 *= rhs;
        self.m31 *= rhs; self.m32 *= rhs; self.m33 *= rhs;
    }
}

impl Mul for Mat3 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            m11: self.m11 * rhs.m11 + self.m12 * rhs.m21 + self.m13 * rhs.m31,
            m12: self.m11 * rhs.m12 + self.m12 * rhs.m22 + self.m13 * rhs.m32,
            m13: self.m11 * rhs.m13 + self.m12 * rhs.m23 + self.m13 * rhs.m33,

            m21: self.m21 * rhs.m11 + self.m22 * rhs.m21 + self.m23 * rhs.m31,
            m22: self.m21 * rhs.m12 + self.m22 * rhs.m22 + self.m23 * rhs.m32,
            m23: self.m21 * rhs.m13 + self.m22 * rhs.m23 + self.m23 * rhs.m33,

            m31: self.m31 * rhs.m11 + self.m32 * rhs.m21 + self.m33 * rhs.m31,
            m32: self.m31 * rhs.m12 + self.m32 * rhs.m22 + self.m33 * rhs.m32,
            m33: self.m31 * rhs.m13 + self.m32 * rhs.m23 + self.m33 * rhs.m33,
        }
    }
}

// ------------------------------------------------------------

impl Add for Mat4 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            m11: self.m11 + rhs.m11, m12: self.m12 + rhs.m12, m13: self.m13 + rhs.m13, m14: self.m14 + rhs.m14,
            m21: self.m21 + rhs.m21, m22: self.m22 + rhs.m22, m23: self.m23 + rhs.m23, m24: self.m24 + rhs.m24,
            m31: self.m31 + rhs.m31, m32: self.m32 + rhs.m32, m33: self.m33 + rhs.m33, m34: self.m34 + rhs.m34,
            m41: self.m41 + rhs.m41, m42: self.m42 + rhs.m42, m43: self.m43 + rhs.m43, m44: self.m44 + rhs.m44,
        }
    }
}

impl AddAssign for Mat4 {
    fn add_assign(&mut self, rhs: Self) {
        self.m11 += rhs.m11; self.m12 += rhs.m12; self.m13 += rhs.m13; self.m14 += rhs.m14;
        self.m21 += rhs.m21; self.m22 += rhs.m22; self.m23 += rhs.m23; self.m24 += rhs.m24;
        self.m31 += rhs.m31; self.m32 += rhs.m32; self.m33 += rhs.m33; self.m34 += rhs.m34;
        self.m41 += rhs.m41; self.m42 += rhs.m42; self.m43 += rhs.m43; self.m44 += rhs.m44;
    }
}

impl Sub for Mat4 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            m11: self.m11 - rhs.m11, m12: self.m12 - rhs.m12, m13: self.m13 - rhs.m13, m14: self.m14 - rhs.m14,
            m21: self.m21 - rhs.m21, m22: self.m22 - rhs.m22, m23: self.m23 - rhs.m23, m24: self.m24 - rhs.m24,
            m31: self.m31 - rhs.m31, m32: self.m32 - rhs.m32, m33: self.m33 - rhs.m33, m34: self.m34 - rhs.m34,
            m41: self.m41 - rhs.m41, m42: self.m42 - rhs.m42, m43: self.m43 - rhs.m43, m44: self.m44 - rhs.m44,
        }
    }
}

impl SubAssign for Mat4 {
    fn sub_assign(&mut self, rhs: Self) {
        self.m11 -= rhs.m11; self.m12 -= rhs.m12; self.m13 -= rhs.m13; self.m14 -= rhs.m14;
        self.m21 -= rhs.m21; self.m22 -= rhs.m22; self.m23 -= rhs.m23; self.m24 -= rhs.m24;
        self.m31 -= rhs.m31; self.m32 -= rhs.m32; self.m33 -= rhs.m33; self.m34 -= rhs.m34;
        self.m41 -= rhs.m41; self.m42 -= rhs.m42; self.m43 -= rhs.m43; self.m44 -= rhs.m44;
    }
}

impl Mul<f64> for Mat4 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            m11: self.m11 * rhs, m12: self.m12 * rhs, m13: self.m13 * rhs, m14: self.m14 * rhs,
            m21: self.m21 * rhs, m22: self.m22 * rhs, m23: self.m23 * rhs, m24: self.m24 * rhs,
            m31: self.m31 * rhs, m32: self.m32 * rhs, m33: self.m33 * rhs, m34: self.m34 * rhs,
            m41: self.m41 * rhs, m42: self.m42 * rhs, m43: self.m43 * rhs, m44: self.m44 * rhs,
        }
    }
}

impl MulAssign<f64> for Mat4 {
    fn mul_assign(&mut self, rhs: f64) {
        self.m11 *= rhs; self.m12 *= rhs; self.m13 *= rhs; self.m14 *= rhs;
        self.m21 *= rhs; self.m22 *= rhs; self.m23 *= rhs; self.m24 *= rhs;
        self.m31 *= rhs; self.m32 *= rhs; self.m33 *= rhs; self.m34 *= rhs;
        self.m41 *= rhs; self.m42 *= rhs; self.m43 *= rhs; self.m44 *= rhs;
    }
}

impl Mul for Mat4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            m11: self.m11 * rhs.m11 + self.m12 * rhs.m21 + self.m13 * rhs.m31 + self.m14 * rhs.m41,
            m12: self.m11 * rhs.m12 + self.m12 * rhs.m22 + self.m13 * rhs.m32 + self.m14 * rhs.m42,
            m13: self.m11 * rhs.m13 + self.m12 * rhs.m23 + self.m13 * rhs.m33 + self.m14 * rhs.m43,
            m14: self.m11 * rhs.m14 + self.m12 * rhs.m24 + self.m13 * rhs.m34 + self.m14 * rhs.m44,

            m21: self.m21 * rhs.m11 + self.m22 * rhs.m21 + self.m23 * rhs.m31 + self.m24 * rhs.m41,
            m22: self.m21 * rhs.m12 + self.m22 * rhs.m22 + self.m23 * rhs.m32 + self.m24 * rhs.m42,
            m23: self.m21 * rhs.m13 + self.m22 * rhs.m23 + self.m23 * rhs.m33 + self.m24 * rhs.m43,
            m24: self.m21 * rhs.m14 + self.m22 * rhs.m24 + self.m23 * rhs.m34 + self.m24 * rhs.m44,

            m31: self.m31 * rhs.m11 + self.m32 * rhs.m21 + self.m33 * rhs.m31 + self.m34 * rhs.m41,
            m32: self.m31 * rhs.m12 + self.m32 * rhs.m22 + self.m33 * rhs.m32 + self.m34 * rhs.m42,
            m33: self.m31 * rhs.m13 + self.m32 * rhs.m23 + self.m33 * rhs.m33 + self.m34 * rhs.m43,
            m34: self.m31 * rhs.m14 + self.m32 * rhs.m24 + self.m33 * rhs.m34 + self.m34 * rhs.m44,

            m41: self.m41 * rhs.m11 + self.m42 * rhs.m21 + self.m43 * rhs.m31 + self.m44 * rhs.m41,
            m42: self.m41 * rhs.m12 + self.m42 * rhs.m22 + self.m43 * rhs.m32 + self.m44 * rhs.m42,
            m43: self.m41 * rhs.m13 + self.m42 * rhs.m23 + self.m43 * rhs.m33 + self.m44 * rhs.m43,
            m44: self.m41 * rhs.m14 + self.m42 * rhs.m24 + self.m43 * rhs.m34 + self.m44 * rhs.m44,
        }
    }
}

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod matrix_tests {
    use super::*;
    use crate::{Vec2, Vec3, Vec4};
    use crate::consts::PI;

    #[test]
    fn mat2_identity_mul_vec() {
        let m = Mat2::identity();
        let v = Vec2::new(3.0, -4.0);
        let r = m.mul_vec2(v);
        assert_eq!(r, v);
    }

    #[test]
    fn mat2_rotation_90() {
        let m = Mat2::rotation(PI / 2.0);
        let v = Vec2::new(1.0, 0.0);
        let r = m.mul_vec2(v);
        assert!((r.x - 0.0).abs() < 1e-10);
        assert!((r.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mat2_inverse() {
        let m = Mat2::new(2.0, 0.0, 0.0, 3.0);
        let inv = m.inverse().unwrap();
        let id = m * inv;
        assert!((id.m11 - 1.0).abs() < 1e-10);
        assert!((id.m22 - 1.0).abs() < 1e-10);
        assert!(id.m12.abs() < 1e-10);
        assert!(id.m21.abs() < 1e-10);
    }

    #[test]
    fn mat3_identity_mul_vec() {
        let m = Mat3::identity();
        let v = Vec3::new(1.0, 2.0, 3.0);
        let r = m.mul_vec3(v);
        assert_eq!(r, v);
    }

    #[test]
    fn mat3_rotation_z_90() {
        let m = Mat3::rotation_z(PI / 2.0);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let r = m.mul_vec3(v);
        assert!((r.x - 0.0).abs() < 1e-10);
        assert!((r.y - 1.0).abs() < 1e-10);
        assert!((r.z - 0.0).abs() < 1e-10);
    }

    #[test]
    fn mat3_inverse() {
        let m = Mat3::scale(2.0, 3.0, 4.0);
        let inv = m.inverse().unwrap();
        let id = m * inv;
        assert!((id.m11 - 1.0).abs() < 1e-10);
        assert!((id.m22 - 1.0).abs() < 1e-10);
        assert!((id.m33 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mat4_identity_mul_vec4() {
        let m = Mat4::identity();
        let v = Vec4::new(1.0, 2.0, 3.0, 1.0);
        let r = m.mul_vec4(v);
        assert_eq!(r, v);
    }

    #[test]
    fn mat4_translation_point() {
        let m = Mat4::translation(10.0, -5.0, 2.0);
        let p = Vec3::new(1.0, 2.0, 3.0);
        let r = m.mul_vec3_point(p);
        assert!((r.x - 11.0).abs() < 1e-10);
        assert!((r.y - -3.0).abs() < 1e-10);
        assert!((r.z - 5.0).abs() < 1e-10);
    }

    #[test]
    fn mat4_scale_direction() {
        let m = Mat4::scale(2.0, 3.0, 4.0);
        let d = Vec3::new(1.0, 1.0, 1.0);
        let r = m.mul_vec3_direction(d);
        assert!((r.x - 2.0).abs() < 1e-10);
        assert!((r.y - 3.0).abs() < 1e-10);
        assert!((r.z - 4.0).abs() < 1e-10);
    }

    #[test]
    fn mat4_perspective_basic() {
        let m = Mat4::perspective(PI / 2.0, 1.0, 0.1, 100.0);
        // Just sanity check some values
        assert!(m.m11 > 0.0);
        assert!(m.m22 > 0.0);
        assert!(m.m33 < 0.0);
        assert!(m.m43 < 0.0);
    }

    #[test]
    fn mat4_orthographic_basic() {
        let m = Mat4::orthographic(-1.0, 1.0, -1.0, 1.0, 0.1, 10.0);
        // Sanity checks
        assert!((m.m11 - 1.0).abs() < 1e-10);
        assert!((m.m22 - 1.0).abs() < 1e-10);
    }
}
// ============================================================
//  QUATERNION MODULE (Quat)
// ============================================================
//
// Design goals:
//  - Represent 3D rotations using unit quaternions
//  - Work nicely with existing Vec3 / Mat3 / Mat4
//  - Provide:
//      * constructors (identity, from_axis_angle, from_euler, from_mat3/mat4)
//      * basic ops (add, sub, mul, scalar mul/div, normalize, inverse)
//      * rotation of Vec3
//      * slerp / nlerp
//      * conversion to Mat3 / Mat4
//
//  Convention:
//    - Quat { w, x, y, z }
//    - Rotation of vector v: q * v * q.conjugate()
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quat {
    // --------------------------------------------------------
    // Constructors
    // --------------------------------------------------------

    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    pub fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    pub fn zero() -> Self {
        Self {
            w: 0.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    pub fn from_scalar_and_vector(w: f64, v: crate::Vec3) -> Self {
        Self {
            w,
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }

    pub fn from_axis_angle(axis: crate::Vec3, angle: f64) -> Self {
        let axis = axis.normalized();
        let half = angle * 0.5;
        let (s, c) = half.sin_cos();
        Self {
            w: c,
            x: axis.x * s,
            y: axis.y * s,
            z: axis.z * s,
        }
    }

    /// From Euler angles (yaw, pitch, roll) in radians.
    /// yaw   = rotation around Y
    /// pitch = rotation around X
    /// roll  = rotation around Z
    pub fn from_euler_yaw_pitch_roll(yaw: f64, pitch: f64, roll: f64) -> Self {
        let (sy, cy) = (yaw * 0.5).sin_cos();
        let (sp, cp) = (pitch * 0.5).sin_cos();
        let (sr, cr) = (roll * 0.5).sin_cos();

        let w = cr * cp * cy + sr * sp * sy;
        let x = sr * cp * cy - cr * sp * sy;
        let y = cr * sp * cy + sr * cp * sy;
        let z = cr * cp * sy - sr * sp * cy;

        Self { w, x, y, z }
    }

    pub fn from_vec3(v: crate::Vec3) -> Self {
        Self {
            w: 0.0,
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }

    // --------------------------------------------------------
    // Basic properties
    // --------------------------------------------------------

    pub fn length(self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn length_squared(self) -> f64 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len == 0.0 {
            Self::identity()
        } else {
            self / len
        }
    }

    pub fn conjugate(self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    pub fn inverse(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq == 0.0 {
            Self::identity()
        } else {
            self.conjugate() / len_sq
        }
    }

    pub fn dot(self, other: Self) -> f64 {
        self.w * other.w
            + self.x * other.x
            + self.y * other.y
            + self.z * other.z
    }

    // --------------------------------------------------------
    // Rotation helpers
    // --------------------------------------------------------

    pub fn rotate_vec3(self, v: crate::Vec3) -> crate::Vec3 {
        // q * v * q^-1
        let qv = Quat::from_vec3(v);
        let res = self * qv * self.inverse();
        crate::Vec3::new(res.x, res.y, res.z)
    }

    pub fn to_mat3(self) -> crate::Mat3 {
        let q = self.normalized();
        let xx = q.x * q.x;
        let yy = q.y * q.y;
        let zz = q.z * q.z;
        let xy = q.x * q.y;
        let xz = q.x * q.z;
        let yz = q.y * q.z;
        let wx = q.w * q.x;
        let wy = q.w * q.y;
        let wz = q.w * q.z;

        crate::Mat3::new(
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),

            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),

            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        )
    }

    pub fn to_mat4(self) -> crate::Mat4 {
        let m3 = self.to_mat3();
        crate::Mat4::from_rows(
            (m3.m11, m3.m12, m3.m13, 0.0),
            (m3.m21, m3.m22, m3.m23, 0.0),
            (m3.m31, m3.m32, m3.m33, 0.0),
            (0.0,    0.0,    0.0,    1.0),
        )
    }

    /// Convert to axis-angle representation.
    /// Returns (axis, angle).
    pub fn to_axis_angle(self) -> (crate::Vec3, f64) {
        let q = self.normalized();
        let angle = 2.0 * q.w.acos();
        let s = (1.0 - q.w * q.w).sqrt();

        if s < 1e-8 {
            (crate::Vec3::new(1.0, 0.0, 0.0), angle)
        } else {
            (
                crate::Vec3::new(q.x / s, q.y / s, q.z / s),
                angle,
            )
        }
    }

    /// Convert to Euler angles (yaw, pitch, roll).
    /// yaw   = rotation around Y
    /// pitch = rotation around X
    /// roll  = rotation around Z
    pub fn to_euler_yaw_pitch_roll(self) -> (f64, f64, f64) {
        let q = self.normalized();

        // yaw (Y axis)
        let siny_cosp = 2.0 * (q.w * q.y + q.z * q.x);
        let cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        let yaw = siny_cosp.atan2(cosy_cosp);

        // pitch (X axis)
        let sinp = 2.0 * (q.w * q.x - q.z * q.y);
        let pitch = if sinp.abs() >= 1.0 {
            sinp.signum() * std::f64::consts::FRAC_PI_2
        } else {
            sinp.asin()
        };

        // roll (Z axis)
        let sinr_cosp = 2.0 * (q.w * q.z + q.x * q.y);
        let cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
        let roll = sinr_cosp.atan2(cosr_cosp);

        (yaw, pitch, roll)
    }

    // --------------------------------------------------------
    // Interpolation
    // --------------------------------------------------------

    /// Normalized linear interpolation (nlerp).
    /// t in [0, 1].
    pub fn nlerp(a: Self, b: Self, t: f64) -> Self {
        let mut b_adj = b;

        // Ensure shortest path
        if a.dot(b) < 0.0 {
            b_adj = Self {
                w: -b.w,
                x: -b.x,
                y: -b.y,
                z: -b.z,
            };
        }

        let w = a.w + (b_adj.w - a.w) * t;
        let x = a.x + (b_adj.x - a.x) * t;
        let y = a.y + (b_adj.y - a.y) * t;
        let z = a.z + (b_adj.z - a.z) * t;

        Self { w, x, y, z }.normalized()
    }

    /// Spherical linear interpolation (slerp).
    /// t in [0, 1].
    pub fn slerp(a: Self, b: Self, t: f64) -> Self {
        let mut b_adj = b;
        let mut cos_theta = a.dot(b);

        // If cos_theta < 0, the interpolation will take the long way around the sphere.
        // To fix this, one quat must be negated.
        if cos_theta < 0.0 {
            b_adj = Self {
                w: -b.w,
                x: -b.x,
                y: -b.y,
                z: -b.z,
            };
            cos_theta = -cos_theta;
        }

        // If the quaternions are very close, fall back to nlerp
        if cos_theta > 0.9995 {
            return Self::nlerp(a, b_adj, t);
        }

        let theta = cos_theta.acos();
        let sin_theta = theta.sin();

        let w1 = ((1.0 - t) * theta).sin() / sin_theta;
        let w2 = (t * theta).sin() / sin_theta;

        Self {
            w: a.w * w1 + b_adj.w * w2,
            x: a.x * w1 + b_adj.x * w2,
            y: a.y * w1 + b_adj.y * w2,
            z: a.z * w1 + b_adj.z * w2,
        }
    }

    // --------------------------------------------------------
    // Random rotations
    // --------------------------------------------------------

    /// Uniform random unit quaternion (uniform rotation).
    pub fn random_unit() -> Self {
        // Using method from "Uniform Random Rotations", Ken Shoemake
        let u1 = crate::rand_f64();
        let u2 = crate::rand_f64() * std::f64::consts::TAU;
        let u3 = crate::rand_f64() * std::f64::consts::TAU;

        let sqrt1_minus_u1 = (1.0 - u1).sqrt();
        let sqrt_u1 = u1.sqrt();

        let w = sqrt1_minus_u1 * u2.cos();
        let x = sqrt1_minus_u1 * u2.sin();
        let y = sqrt_u1 * u3.cos();
        let z = sqrt_u1 * u3.sin();

        Self { w, x, y, z }
    }

    // --------------------------------------------------------
    // Utility
    // --------------------------------------------------------

    pub fn is_finite(self) -> bool {
        self.w.is_finite() && self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    pub fn approx_eq(self, other: Self, eps: f64) -> bool {
        (self.w - other.w).abs() < eps
            && (self.x - other.x).abs() < eps
            && (self.y - other.y).abs() < eps
            && (self.z - other.z).abs() < eps
    }
}

// ============================================================
//  Operators
// ============================================================

impl Add for Quat {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            w: self.w + rhs.w,
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl AddAssign for Quat {
    fn add_assign(&mut self, rhs: Self) {
        self.w += rhs.w;
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub for Quat {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            w: self.w - rhs.w,
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl SubAssign for Quat {
    fn sub_assign(&mut self, rhs: Self) {
        self.w -= rhs.w;
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Mul<f64> for Quat {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Self {
            w: self.w * rhs,
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl MulAssign<f64> for Quat {
    fn mul_assign(&mut self, rhs: f64) {
        self.w *= rhs;
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl Div<f64> for Quat {
    type Output = Self;

    fn div(self, rhs: f64) -> Self {
        Self {
            w: self.w / rhs,
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl DivAssign<f64> for Quat {
    fn div_assign(&mut self, rhs: f64) {
        self.w /= rhs;
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

impl Neg for Quat {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            w: -self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Mul for Quat {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        // Hamilton product
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }
}

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod quat_tests {
    use super::*;
    use crate::{Vec3, Mat3};
    use crate::consts::PI;

    #[test]
    fn quat_identity_rotation() {
        let q = Quat::identity();
        let v = Vec3::new(1.0, 2.0, 3.0);
        let r = q.rotate_vec3(v);
        assert_eq!(r, v);
    }

    #[test]
    fn quat_axis_angle_90_deg_x() {
        let axis = Vec3::new(1.0, 0.0, 0.0);
        let q = Quat::from_axis_angle(axis, PI / 2.0);
        let v = Vec3::new(0.0, 1.0, 0.0);
        let r = q.rotate_vec3(v);
        assert!((r.x - 0.0).abs() < 1e-10);
        assert!((r.y - 0.0).abs() < 1e-10);
        assert!((r.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn quat_axis_angle_180_deg_y() {
        let axis = Vec3::new(0.0, 1.0, 0.0);
        let q = Quat::from_axis_angle(axis, PI);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let r = q.rotate_vec3(v);
        assert!((r.x + 1.0).abs() < 1e-10);
        assert!((r.y - 0.0).abs() < 1e-10);
        assert!((r.z - 0.0).abs() < 1e-10);
    }

    #[test]
    fn quat_length_and_normalize() {
        let q = Quat::new(2.0, 0.0, 0.0, 0.0);
        assert!((q.length() - 2.0).abs() < 1e-10);
        let n = q.normalized();
        assert!((n.length() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn quat_inverse() {
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let q = Quat::from_axis_angle(axis, PI / 3.0);
        let q_inv = q.inverse();
        let id = q * q_inv;
        assert!((id.w - 1.0).abs() < 1e-10);
        assert!(id.x.abs() < 1e-10);
        assert!(id.y.abs() < 1e-10);
        assert!(id.z.abs() < 1e-10);
    }

    #[test]
    fn quat_dot_and_negation() {
        let q = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 1.0);
        let q_neg = -q;
        assert!((q.dot(q_neg) + q.length_squared()).abs() < 1e-10);
    }

    #[test]
    fn quat_to_mat3_and_back() {
        let axis = Vec3::new(0.0, 1.0, 0.0);
        let q = Quat::from_axis_angle(axis, PI / 4.0);
        let m = q.to_mat3();

        // Rotate vector using both
        let v = Vec3::new(1.0, 0.0, 0.0);
        let r1 = q.rotate_vec3(v);
        let r2 = m.mul_vec3(v);

        assert!((r1.x - r2.x).abs() < 1e-10);
        assert!((r1.y - r2.y).abs() < 1e-10);
        assert!((r1.z - r2.z).abs() < 1e-10);
    }

    #[test]
    fn quat_slerp_midpoint() {
        let q1 = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.0);
        let q2 = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), PI);
        let qm = Quat::slerp(q1, q2, 0.5);

        let (_, angle) = qm.to_axis_angle();
        assert!((angle - PI / 2.0).abs() < 1e-6);
    }

    #[test]
    fn quat_nlerp_close_to_slerp() {
        let q1 = Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 0.0);
        let q2 = Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), PI / 2.0);

        let t = 0.3;
        let qs = Quat::slerp(q1, q2, t);
        let qn = Quat::nlerp(q1, q2, t);

        assert!(qs.approx_eq(qn, 1e-2));
    }

    #[test]
    fn quat_random_unit_is_unit() {
        for _ in 0..100 {
            let q = Quat::random_unit();
            let len = q.length();
            assert!((len - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn quat_to_from_euler() {
        let yaw = 0.3;
        let pitch = -0.7;
        let roll = 1.2;

        let q = Quat::from_euler_yaw_pitch_roll(yaw, pitch, roll);
        let (ryaw, rpitch, rroll) = q.to_euler_yaw_pitch_roll();

        assert!((yaw - ryaw).abs() < 1e-5);
        assert!((pitch - rpitch).abs() < 1e-5);
        assert!((roll - rroll).abs() < 1e-5);
    }

    #[test]
    fn quat_rotate_vec_matches_axis_angle() {
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let angle = PI / 2.0;
        let q = Quat::from_axis_angle(axis, angle);

        let v = Vec3::new(1.0, 0.0, 0.0);
        let r = q.rotate_vec3(v);

        assert!((r.x - 0.0).abs() < 1e-10);
        assert!((r.y - 1.0).abs() < 1e-10);
        assert!((r.z - 0.0).abs() < 1e-10);
    }

    #[test]
    fn quat_is_finite() {
        let q = Quat::new(1.0, 2.0, 3.0, 4.0);
        assert!(q.is_finite());
        let q_nan = Quat::new(f64::NAN, 0.0, 0.0, 0.0);
        assert!(!q_nan.is_finite());
    }
}
// ============================================================
//  COMPLEX NUMBER MODULE (Complex)
// ============================================================
//
//  Design goals:
//   - f64-based complex numbers
//   - Full operator support (+, -, *, /, +=, -=, *=, /=, Neg)
//   - Conversions to/from polar form
//   - Magnitude, phase, normalization
//   - Complex exponential, log, pow
//   - Roots (nth roots)
//   - Trigonometric functions (sin, cos, tan, etc.)
//   - Hyperbolic functions
//   - Random complex numbers
//   - Tests
//
//  Representation:
//      Complex { re, im }
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    // --------------------------------------------------------
    // Constructors
    // --------------------------------------------------------

    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    pub fn i() -> Self {
        Self { re: 0.0, im: 1.0 }
    }

    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    pub fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    pub fn imag(im: f64) -> Self {
        Self { re: 0.0, im }
    }

    // --------------------------------------------------------
    // Basic properties
    // --------------------------------------------------------

    pub fn magnitude(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn magnitude_squared(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn phase(self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn normalized(self) -> Self {
        let mag = self.magnitude();
        if mag == 0.0 {
            Self::zero()
        } else {
            Self {
                re: self.re / mag,
                im: self.im / mag,
            }
        }
    }

    pub fn conjugate(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn inverse(self) -> Self {
        let denom = self.magnitude_squared();
        if denom == 0.0 {
            Self::zero()
        } else {
            Self {
                re: self.re / denom,
                im: -self.im / denom,
            }
        }
    }

    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    // --------------------------------------------------------
    // Arithmetic
    // --------------------------------------------------------

    pub fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    pub fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    pub fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    pub fn div(self, other: Self) -> Self {
        let denom = other.magnitude_squared();
        if denom == 0.0 {
            return Self::zero();
        }
        Self {
            re: (self.re * other.re + self.im * other.im) / denom,
            im: (self.im * other.re - self.re * other.im) / denom,
        }
    }

    // --------------------------------------------------------
    // Exponential & Logarithm
    // --------------------------------------------------------

    pub fn exp(self) -> Self {
        let e = self.re.exp();
        Self {
            re: e * self.im.cos(),
            im: e * self.im.sin(),
        }
    }

    pub fn ln(self) -> Self {
        Self {
            re: self.magnitude().ln(),
            im: self.phase(),
        }
    }

    pub fn pow(self, other: Self) -> Self {
        // z^w = exp(w * ln(z))
        (other * self.ln()).exp()
    }

    pub fn powf(self, f: f64) -> Self {
        // z^f = exp(f * ln(z))
        (self.ln() * f).exp()
    }

    // --------------------------------------------------------
    // Roots
    // --------------------------------------------------------

    pub fn sqrt(self) -> Self {
        let r = self.magnitude();
        let theta = self.phase() / 2.0;
        Self::from_polar(r.sqrt(), theta)
    }

    pub fn nth_roots(self, n: u32) -> Vec<Self> {
        let mut roots = Vec::new();
        let r = self.magnitude().powf(1.0 / n as f64);
        let theta = self.phase();

        for k in 0..n {
            let angle = (theta + 2.0 * std::f64::consts::PI * k as f64) / n as f64;
            roots.push(Self::from_polar(r, angle));
        }

        roots
    }

    // --------------------------------------------------------
    // Trigonometric functions
    // --------------------------------------------------------

    pub fn sin(self) -> Self {
        Self {
            re: self.re.sin() * self.im.cosh(),
            im: self.re.cos() * self.im.sinh(),
        }
    }

    pub fn cos(self) -> Self {
        Self {
            re: self.re.cos() * self.im.cosh(),
            im: -self.re.sin() * self.im.sinh(),
        }
    }

    pub fn tan(self) -> Self {
        self.sin() / self.cos()
    }

    // --------------------------------------------------------
    // Hyperbolic functions
    // --------------------------------------------------------

    pub fn sinh(self) -> Self {
        Self {
            re: self.re.sinh() * self.im.cos(),
            im: self.re.cosh() * self.im.sin(),
        }
    }

    pub fn cosh(self) -> Self {
        Self {
            re: self.re.cosh() * self.im.cos(),
            im: self.re.sinh() * self.im.sin(),
        }
    }

    pub fn tanh(self) -> Self {
        self.sinh() / self.cosh()
    }

    // --------------------------------------------------------
    // Random complex numbers
    // --------------------------------------------------------

    pub fn random_unit() -> Self {
        let angle = crate::rand_f64_range(0.0, std::f64::consts::TAU);
        Self::from_polar(1.0, angle)
    }

    pub fn random_range(min: f64, max: f64) -> Self {
        Self {
            re: crate::rand_f64_range(min, max),
            im: crate::rand_f64_range(min, max),
        }
    }

    // --------------------------------------------------------
    // Utility
    // --------------------------------------------------------

    pub fn approx_eq(self, other: Self, eps: f64) -> bool {
        (self.re - other.re).abs() < eps && (self.im - other.im).abs() < eps
    }
}

// ============================================================
//  Operators
// ============================================================

impl Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        self.add(rhs)
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self.sub(rhs)
    }
}

impl SubAssign for Complex {
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.mul(rhs)
    }
}

impl MulAssign for Complex {
    fn mul_assign(&mut self, rhs: Self) {
        let r = self.mul(rhs);
        self.re = r.re;
        self.im = r.im;
    }
}

impl Div for Complex {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self.div(rhs)
    }
}

impl DivAssign for Complex {
    fn div_assign(&mut self, rhs: Self) {
        let r = self.div(rhs);
        self.re = r.re;
        self.im = r.im;
    }
}

impl Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl MulAssign<f64> for Complex {
    fn mul_assign(&mut self, rhs: f64) {
        self.re *= rhs;
        self.im *= rhs;
    }
}

impl Div<f64> for Complex {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self {
            re: self.re / rhs,
            im: self.im / rhs,
        }
    }
}

impl DivAssign<f64> for Complex {
    fn div_assign(&mut self, rhs: f64) {
        self.re /= rhs;
        self.im /= rhs;
    }
}

impl Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod complex_tests {
    use super::*;
    use crate::consts::PI;

    #[test]
    fn complex_basic_add() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, -1.0);
        let r = a + b;
        assert_eq!(r, Complex::new(4.0, 1.0));
    }

    #[test]
    fn complex_basic_mul() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let r = a * b;
        assert_eq!(r, Complex::new(-5.0, 10.0));
    }

    #[test]
    fn complex_magnitude_phase() {
        let c = Complex::new(3.0, 4.0);
        assert!((c.magnitude() - 5.0).abs() < 1e-10);
        assert!((c.phase() - (4.0f64).atan2(3.0)).abs() < 1e-10);
    }

    #[test]
    fn complex_exp_ln() {
        let c = Complex::new(1.0, 1.0);
        let e = c.exp();
        let l = e.ln();
        assert!(c.approx_eq(l, 1e-10));
    }

    #[test]
    fn complex_pow() {
        let c = Complex::new(2.0, 3.0);
        let r = c.powf(2.0);
        let expected = c * c;
        assert!(r.approx_eq(expected, 1e-10));
    }

    #[test]
    fn complex_sqrt() {
        let c = Complex::new(3.0, 4.0);
        let r = c.sqrt();
        assert!((r * r).approx_eq(c, 1e-10));
    }

    #[test]
    fn complex_trig() {
        let c = Complex::new(1.0, 0.5);
        let s = c.sin();
        let c2 = c.cos();
        let t = c.tan();
        assert!(s.is_finite());
        assert!(c2.is_finite());
        assert!(t.is_finite());
    }

    #[test]
    fn complex_random_unit() {
        for _ in 0..100 {
            let c = Complex::random_unit();
            assert!((c.magnitude() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn complex_nth_roots() {
        let c = Complex::new(1.0, 0.0);
        let roots = c.nth_roots(4);
        assert_eq!(roots.len(), 4);
        for r in roots {
            assert!((r * r * r * r).approx_eq(c, 1e-10));
        }
    }
}
// ============================================================
//  SIMD MODULE (Vec2Simd, Vec3Simd, Vec4Simd, BatchOps)
// ============================================================
//
//  Uses Rust's stable `std::simd` API.
