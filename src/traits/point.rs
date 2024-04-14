/// Traits that the Point type should implement for use by vectune::Builder.
///
/// # Examples
///
/// ```ignore
/// #[derive(Serialize, Deserialize, Clone, Debug)]
/// struct Point(Vec<f32>);
/// impl PointInterface for Point {
///     ...
/// }
/// ```
pub trait PointInterface: Clone + Sync {
  /// A function that returns the distance between two Points. Typically, the Euclidean distance is used.
  ///
  /// # Examples
  ///
  /// ```ignore
  /// fn distance(&self, other: &Self) -> f32 {
  ///     self.0
  ///         .iter()
  ///         .zip(other.0.iter())
  ///         .map(|(a, b)| {
  ///             let c = a - b;
  ///             c * c
  ///         })
  ///         .sum::<f32>()
  ///         .sqrt()
  /// }
  /// ```
  fn distance(&self, other: &Self) -> f32;

  /// The number of dimensions of a vector.
  fn dim() -> u32;

  /// Addition of two Points. Used by Builder to find the centroid.
  ///
  ///　# Examples
  ///
  /// ```ignore
  /// fn add(&self, other: &Self) -> Self {
  ///     Point::from_f32_vec(
  ///         self.to_f32_vec()
  ///             .into_iter()
  ///             .zip(other.to_f32_vec().into_iter())
  ///             .map(|(x, y)| x + y)
  ///             .collect(),
  ///     )
  /// }
  /// ```
  fn add(&self, other: &Self) -> Self;

  /// Division of a Point. Used by Builder to find the centroid.
  ///
  ///　# Examples
  ///
  /// ```ignore
  /// fn div(&self, divisor: &usize) -> Self {
  ///     Point::from_f32_vec(
  ///         self.to_f32_vec()
  ///             .into_iter()
  ///             .map(|v| v / *divisor as f32)
  ///             .collect(),
  ///     )
  /// }
  /// ```
  fn div(&self, divisor: &usize) -> Self;
}