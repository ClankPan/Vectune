
use rand::distributions::{Distribution, WeightedIndex};
/*
Note:
  - kmeans crate uses SIMD intrinsics from the packed_simd crate. To run on wasm which does not support SIMD, I implement it by my self.
*/
use rand::rngs::SmallRng;
use rand::Rng;

use crate::Point;


pub struct PQ<P> {
  rng: SmallRng,
  d: usize,
  d_: usize,
  m: usize,
  k: usize,
  vectors: Vec<P>,
}

impl<P> PQ<P>
where
    P: Point,
{
  pub fn new(rng: SmallRng, m: usize, k: usize, d: usize, vectors: Vec<P>) -> Self {
    assert!(d % m == 0);
    assert!(k % 2 == 0);
    assert!(k < vectors.len());
  
    let d_ = d / m;

    Self {
      rng,
      d,
      d_,
      m,
      k,
      vectors
    }
  }

  pub fn quantize(mut self) -> (Vec<(P, Vec<usize>)>, Vec<Vec<P>>) {
    let mut codebooks: Vec<Vec<P>> = Vec::new();
    let mut quantums: Vec<Vec<usize>> = vec![vec![]; self.vectors.len()];

    println!("params:: m: {}, d: {}, k: {}, d_: {}", &self.m,  &self.d,  &self.k,  &self.d_);

    for sub_i in 0..self.m {
      println!("sub_i: {}", sub_i);
      let start_i = sub_i * self.d_;
      let end_i = start_i + self.d_;
      println!("start_i, end_i: {} {}", start_i, end_i);

      let subvectors: Vec<P> = self.vectors.iter().map(|p| {
        // println!("p.to_f32_vec()[start_i..end_i].to_vec() {:?}", p.to_f32_vec()[start_i..end_i].to_vec());
        P::from_f32_vec(p.to_f32_vec()[start_i..end_i].to_vec())
      }).collect();

      let initial_centroids = self.random_k_centroids(&subvectors);
      let (sub_codewords, sub_codebook) = self.k_means(initial_centroids, subvectors);
      
      let mut i = 0;
      for codewords in &mut quantums {
        codewords.push(sub_codewords[i]);
        i += 1;
      }
      codebooks.push(sub_codebook);
    }

    (self.vectors.into_iter().zip(quantums).collect(), codebooks)
    
  }

  fn random_k_centroids(&mut self, subvectors: &Vec<P>) -> Vec<P> {

    let mut centroids: Vec<P> = Vec::new();

    // Randomly select the first centroid from the data points
    let random_subvector = &subvectors[self.rng.gen_range(0..self.vectors.len())];
    centroids.push(random_subvector.clone());
    
    while centroids.len() < self.k {
      // Compute the distance from the point to the closest centroid
      let distances: Vec<f32> = subvectors.iter().map(|p| {
        centroids.iter().map(|centroid| {
            p.distance(centroid)
        }).fold(f32::INFINITY, f32::min)
      }).collect();

      // Convert distances to probabilities
      let sum_of_distances: f32 = distances.iter().sum();
      let probabilities: Vec<f32> = distances.iter().map(|&d| d / sum_of_distances).collect();
      // println!("probabilities: {:?}", distances);

      // Select a new centroid based on the computed probabilities
      let wighted_index = WeightedIndex::new(&probabilities);
      // println!("{:?}", wighted_index);
      let dist = wighted_index.unwrap();
      let selected_id = dist.sample(&mut self.rng);
      centroids.push(subvectors[selected_id].clone());
    }

    centroids

  }

  fn find_closest_centroid(point: &P, centroids: &Vec<P>) -> usize {
    centroids.iter()
      .enumerate()
      .map(|(index, centroid)| (index, point.distance(centroid)))
      .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
      .map(|(index, _)| index)
      .unwrap()
  }

  fn update_centroids(&self, points: &Vec<P>, assignments: &[usize]) -> Vec<P> {
    let mut centroids: Vec<Vec<f32>> = vec![vec![0.0; self.d_]; self.k];
    let mut counts = vec![0; self.k];

    for (point, &assignment) in points.iter().zip(assignments.iter()) {
      counts[assignment] += 1;
      for (i, val) in point.to_f32_vec().iter().enumerate() {
        centroids[assignment][i] += val;
      }
    }

    for (centroid, &count) in centroids.iter_mut().zip(counts.iter()) {
      if count > 0 {
        for val in centroid.iter_mut() {
          *val /= count as f32;
        }
      }
    }

    centroids.into_iter().map(|vec| P::from_f32_vec(vec)).collect()
  }

  fn k_means(&self, initial_centroids: Vec<P>, subvectors: Vec<P>) -> (Vec<usize>, Vec<P>){
    let mut centroids: Vec<P> = initial_centroids;

    loop {
      let assignments: Vec<usize> = subvectors.iter()
        .map(|point| Self::find_closest_centroid(point, &centroids))
        .collect();

      let new_centroids = self.update_centroids(&subvectors, &assignments);

      if new_centroids.iter().zip(centroids.iter()).all(|(a, b)| a.to_f32_vec() == b.to_f32_vec()) {
        return (assignments, centroids);
      }

      centroids = new_centroids;
    }
  }
  
}