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


/// Traits that should be implemented for searching, inserting, and deleting after indexing.
///
/// # Exapmles
///
/// ```ignore
/// 
/// struct Graph<P>
/// where
///     P: VPoint,
/// {
///     nodes: Vec<(P, Vec<u32>)>,
///     backlinks: Vec<Vec<u32>>,
///     cemetery: Vec<u32>,
///     centroid: u32,
/// }
/// 
/// impl<P> VGraph<P> for Graph<P>
/// where
///     P: VPoint,
/// {
///     fn alloc(&mut self, point: P) -> u32 {
///         self.nodes.push((point, vec![]));
///         self.backlinks.push(vec![]);
///         (self.nodes.len() - 1) as u32
///     }
/// 
///     fn free(&mut self, _id: &u32) {
///         // todo!()
///     }
/// 
///     fn cemetery(&self) -> Vec<u32> {
///         self.cemetery.clone()
///     }
/// 
///     fn clear_cemetery(&mut self) {
///         self.cemetery = Vec::new();
///     }
/// 
///     fn backlink(&self, id: &u32) -> Vec<u32> {
///         self.backlinks[*id as usize].clone()
///     }
/// 
///     fn get(&mut self, id: &u32) -> (P, Vec<u32>) {
///         let node = &self.nodes[*id as usize];
///         node.clone()
///     }
/// 
///     fn size_l(&self) -> usize {
///         125
///     }
/// 
///     fn size_r(&self) -> usize {
///         70
///     }
/// 
///     fn size_a(&self) -> f32 {
///         2.0
///     }
/// 
///     fn start_id(&self) -> u32 {
///         self.centroid
///     }
/// 
///     fn overwirte_out_edges(&mut self, id: &u32, edges: Vec<u32>) {
///         for out_i in &self.nodes[*id as usize].1 {
///             let backlinks = &mut self.backlink(out_i);
///             backlinks.retain(|out_i| out_i != id)
///         }
/// 
///         for out_i in &edges {
///             let backlinks = &mut self.backlink(out_i);
///             backlinks.push(*id);
///             backlinks.sort();
///             backlinks.dedup();
///         }
/// 
///         self.nodes[*id as usize].1 = edges;
///     }
/// }
/// ```
pub trait GraphInterface<P> {
  fn alloc(&mut self, point: P) -> u32;
  fn free(&mut self, id: &u32);
  fn cemetery(&self) -> Vec<u32>;
  fn clear_cemetery(&mut self);
  fn backlink(&self, id: &u32) -> Vec<u32>;
  fn get(&mut self, id: &u32) -> (P, Vec<u32>);
  fn size_l(&self) -> usize;
  fn size_r(&self) -> usize;
  fn size_a(&self) -> f32;
  fn start_id(&self) -> u32;
  fn overwirte_out_edges(&mut self, id: &u32, edges: Vec<u32>); // backlinkを処理する必要がある。
}