use rand::distributions::{Distribution, WeightedIndex};
/*
Note:
  - kmeans crate uses SIMD intrinsics from the packed_simd crate. To run on wasm which does not support SIMD, I implement it by my self.
*/
use rand::rngs::SmallRng;
use rand::Rng;
use rayon::prelude::*;

use crate::Point;

pub struct KMeans<P> {
    rng: SmallRng,
    k: usize,
    max_iter: usize,
    vectors: Vec<P>,
}

impl<P> KMeans<P>
where
    P: Point,
{
    pub fn new(rng: SmallRng, k: usize, max_iter: usize, vectors: Vec<P>) -> Self {
        assert!(k < vectors.len());

        Self {
            rng,
            k,
            max_iter,
            vectors,
        }
    }

    pub fn kmeans_pp_centroids(&mut self) -> Vec<P> {
        let mut centroids: Vec<P> = Vec::new();

        // Randomly select the first centroid from the data points
        let random_vector = &self.vectors[self.rng.gen_range(0..self.vectors.len())];
        centroids.push(random_vector.clone());

        while centroids.len() < self.k {
            // Compute the distance from the point to the closest centroid
            let distances: Vec<f32> = self
                .vectors
                .par_iter()
                .map(|p| {
                    centroids
                        .iter()
                        .map(|centroid| p.distance(centroid))
                        .fold(f32::INFINITY, f32::min)
                })
                .collect();

            // Convert distances to probabilities
            let sum_of_distances: f32 = distances.iter().sum();
            let probabilities: Vec<f32> = distances.iter().map(|&d| d / sum_of_distances).collect();
            // println!("probabilities: {:?}", distances);

            // Select a new centroid based on the computed probabilities
            let wighted_index = WeightedIndex::new(&probabilities);
            // println!("{:?}", wighted_index);
            let dist = wighted_index.unwrap();
            let selected_id = dist.sample(&mut self.rng);
            centroids.push(self.vectors[selected_id].clone());
        }

        centroids
    }

    pub fn find_closest_centroid(point: &P, centroids: &Vec<P>) -> usize {
        centroids
            .iter()
            .enumerate()
            .map(|(index, centroid)| (index, point.distance(centroid)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap()
    }

    pub fn find_second_closest_centroid(point: &P, centroids: &Vec<P>) -> usize {
        let mut centroids = centroids
            .iter()
            .enumerate()
            .map(|(index, centroid)| (index, point.distance(centroid)))
            .collect::<Vec<_>>();

        centroids.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let (index, _) = centroids[1];
        index
    }

    fn update_centroids(&self, points: &Vec<P>, assignments: &[usize]) -> Vec<P> {
        let mut centroids: Vec<Vec<f32>> = vec![vec![0.0; P::dim() as usize]; self.k];
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

        centroids
            .into_iter()
            .map(|vec| P::from_f32_vec(vec))
            .collect()
    }

    pub fn calculate(self, initial_centroids: Vec<P>) -> (Vec<usize>, Vec<P>) {
        let mut centroids: Vec<P> = initial_centroids;

        let _count = 0;

        loop {
            let assignments: Vec<usize> = self
                .vectors
                .par_iter()
                .map(|point| Self::find_closest_centroid(point, &centroids))
                .collect();

            let new_centroids = self.update_centroids(&self.vectors, &assignments);

            if new_centroids
                .iter()
                .zip(&centroids)
                .all(|(a, b)| a.distance(b) == 0.0)
            {
                println!("k-means end");
                return (assignments, centroids);
            }

            centroids = new_centroids;

            // if count > self.max_iter {
            //   return (assignments, centroids);
            // }
            // count += 1;
        }
    }
}
