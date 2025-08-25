//! # Complex_RS
//!
//! `complex_rs`는 Rust로 작성된 간단하고 효율적인 복소수 라이브-러리입니다.
//! 제네릭을 사용하여 `f32`와 `f64` 타입을 모두 지원하며,
//! 기본적인 복소수 연산을 제공합니다.

// 필요한 트레잇들을 표준 라이브러리와 외부 라이브러리에서 가져옵니다.
use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg};
use num_traits::Float; // f32, f64와 같은 부동소수점 타입들이 구현하는 트레잇

/// 복소수를 나타내는 제네릭 구조체.
///
/// # 제네릭 파라미터
/// * `T`: `f32` 또는 `f64`와 같이 `Float` 트레잇을 구현하는 부동소수점 타입.
///
/// # 예시
/// ```
/// use complex_rs::Complex;
/// let z_f64 = Complex::<f64>::new(3.0, 4.0);
/// let z_f32 = Complex::<f32>::new(1.0, -2.0);
/// ```
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Complex<T: Float> {
    /// 실수부 (Real part)
    pub re: T,
    /// 허수부 (Imaginary part)
    pub im: T,
}

// T가 Float 타입일 때만 이 블록을 구현합니다.
impl<T: Float> Complex<T> {
    /// 새로운 복소수를 생성합니다.
    pub fn new(re: T, im: T) -> Self {
        Self { re, im }
    }

    /// 켤레복소수를 반환합니다. (a + bi -> a - bi)
    pub fn conjugate(&self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// 크기(절댓값, Norm)를 반환합니다. |z| = sqrt(a^2 + b^2)
    pub fn norm(&self) -> T {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    /// 크기의 제곱을 반환합니다. |z|^2 = a^2 + b^2
    /// `norm()`보다 계산이 빠르므로, 크기 비교나 확률 계산에 더 효율적입니다.
    #[inline]
    pub fn magnitude_squared(&self) -> T {
        self.re * self.re + self.im * self.im
    }

    // --- 자주 사용하는 상수들 ---

    /// 복소수 0 (0 + 0i)
    pub fn zero() -> Self {
        Self::new(T::zero(), T::zero())
    }

    /// 복소수 1 (1 + 0i)
    pub fn one() -> Self {
        Self::new(T::one(), T::zero())
    }

    /// 허수 단위 i (0 + 1i)
    pub fn i() -> Self {
        Self::new(T::zero(), T::one())
    }
}

// --- 연산자 오버로딩 ---

impl<T: Float> Add for Complex<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl<T: Float> Sub for Complex<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl<T: Float> Mul for Complex<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

/// 복소수 나눗셈 구현
/// (a + bi) / (c + di) = [(ac + bd) + (bc - ad)i] / (c^2 + d^2)
impl<T: Float> Div for Complex<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.magnitude_squared();
        let re = (self.re * rhs.re + self.im * rhs.im) / denom;
        let im = (self.im * rhs.re - self.re * rhs.im) / denom;
        Self::new(re, im)
    }
}

/// 단항 연산자 '-' 구현 (부호 반전)
impl<T: Float + Neg<Output = T>> Neg for Complex<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

// --- 실수(스칼라)와의 연산 ---

/// Complex<T> * T (복소수 * 실수)
impl<T: Float> Mul<T> for Complex<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

// --- 출력 형식 지정 ---

impl<T: Float + fmt::Display> fmt::Display for Complex<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let abs_im = self.im.abs();
        if self.im < T::zero() {
            write!(f, "{} - {}i", self.re, abs_im)
        } else {
            write!(f, "{} + {}i", self.re, abs_im)
        }
    }
}

// --- 단위 테스트 ---

// 이 모듈은 'cargo test'를 실행할 때만 컴파일됩니다.
#[cfg(test)]
mod tests {
    // 부모 모듈(우리 라이브-러리)의 모든 것을 가져옵니다.
    use super::*;

    // 부동소수점 비교를 위한 작은 허용 오차
    const TOLERANCE: f64 = 1e-10;

    fn assert_complex_eq(a: Complex<f64>, b: Complex<f64>) {
        assert!((a.re - b.re).abs() < TOLERANCE);
        assert!((a.im - b.im).abs() < TOLERANCE);
    }

    #[test]
    fn test_new() {
        let z = Complex::<f64>::new(1.0, 2.0);
        assert_eq!(z.re, 1.0);
        assert_eq!(z.im, 2.0);
    }

    #[test]
    fn test_addition() {
        let z1 = Complex::<f64>::new(2.0, 3.0);
        let z2 = Complex::<f64>::new(1.0, -1.0);
        let expected = Complex::<f64>::new(3.0, 2.0);
        assert_complex_eq(z1 + z2, expected);
    }

    #[test]
    fn test_subtraction() {
        let z1 = Complex::<f64>::new(2.0, 3.0);
        let z2 = Complex::<f64>::new(1.0, -1.0);
        let expected = Complex::<f64>::new(1.0, 4.0);
        assert_complex_eq(z1 - z2, expected);
    }

    #[test]
    fn test_multiplication() {
        let z1 = Complex::<f64>::new(2.0, 3.0);
        let z2 = Complex::<f64>::new(1.0, -1.0);
        let expected = Complex::<f64>::new(5.0, 1.0);
        assert_complex_eq(z1 * z2, expected);
    }

    #[test]
    fn test_division() {
        let z1 = Complex::<f64>::new(5.0, 1.0);
        let z2 = Complex::<f64>::new(1.0, -1.0);
        let expected = Complex::<f64>::new(2.0, 3.0);
        assert_complex_eq(z1 / z2, expected);
    }

    #[test]
    fn test_conjugate() {
        let z = Complex::<f64>::new(3.0, -4.0);
        let expected = Complex::<f64>::new(3.0, 4.0);
        assert_complex_eq(z.conjugate(), expected);
    }

    #[test]
    fn test_magnitude() {
        let z = Complex::<f64>::new(3.0, 4.0);
        assert!((z.norm() - 5.0).abs() < TOLERANCE);
        assert!((z.magnitude_squared() - 25.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_negation() {
        let z = Complex::<f64>::new(2.5, -7.0);
        let expected = Complex::<f64>::new(-2.5, 7.0);
        assert_complex_eq(-z, expected);
    }
}