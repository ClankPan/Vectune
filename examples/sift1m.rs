#![feature(portable_simd)]
use std::simd::f32x4;

use vectune::{Builder as VamanaBuilder, FreshVamanaMap, Point as VamanaPoint};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use serde::{Deserialize, Serialize};

use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use std::io::{self, BufReader};

fn read_fvecs(file_path: &str) -> io::Result<Vec<Vec<f32>>> {
    let file = File::open(file_path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();

    while let Ok(dim) = reader.read_i32::<LittleEndian>() {
        let mut vec = Vec::with_capacity(dim as usize);
        for _ in 0..dim {
            let val = reader.read_f32::<LittleEndian>()?;
            vec.push(val);
        }
        vectors.push(vec);
    }

    Ok(vectors)
}

fn read_ivecs(file_path: &str) -> io::Result<Vec<Vec<i32>>> {
    let file = File::open(file_path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();

    while let Ok(dim) = reader.read_i32::<LittleEndian>() {
        let mut vec = Vec::with_capacity(dim as usize);
        for _ in 0..dim {
            let val = reader.read_i32::<LittleEndian>()?;
            vec.push(val);
        }
        vectors.push(vec);
    }

    Ok(vectors)
}

fn main() {
    let seed: u64 = 11923543545843533243;
    let mut rng = SmallRng::seed_from_u64(seed);

    let base_vectors = read_fvecs("./test_data/sift/sift_base.fvecs").unwrap();

    let mut points = Vec::new();
    let mut values = Vec::new();
    for (i, vec) in base_vectors.iter().enumerate() {
        points.push(Point(vec.to_vec()));
        values.push(i);
    }

    println!("building vamana...");
    let vamana_builder = VamanaBuilder::default();
    // vamana_builder.set_l(250);
    let vamana_map: FreshVamanaMap<Point, usize> =
        vamana_builder.build(points.clone(), values.clone());

    // Search in FreshVamana
    let query_vectors = read_fvecs("./test_data/sift/sift_query.fvecs").unwrap();
    // let query_i = rng.gen_range(0..query_vectors.len() as usize);
    // let query = &query_vectors[query_i];
    // println!("searcing vamana...");
    // let vamana_results = vamana_map.search(&Point(query.to_vec()));
    // println!("{:?}\n\n", vamana_results);

    let groundtruth = read_ivecs("test_data/sift/sift_groundtruth.ivecs").unwrap();
    // println!("groundtruth: {:?}", groundtruth[query_i]);
    // println!(
    //     "results: {:?}",
    //     vamana_results
    //         .iter()
    //         .map(|(_, i)| *i)
    //         .collect::<Vec<usize>>()
    // );





    let round = 100;
    let mut hit = 0;
    // println!("query_vectors len: {:?}", &query_vectors[0..100]);
    for _ in 0..round {
        let query_i = rng.gen_range(0..query_vectors.len() as usize);
        let query = &query_vectors[query_i];

        let vamana_results = vamana_map.search(&Point(query.to_vec()));
        let top5 = &vamana_results
            .into_iter()
            .map(|(_, i)| i as i32)
            .collect::<Vec<i32>>()[0..5];
        let top5_groundtruth = &groundtruth[query_i][0..5];
        // println!("{:?}\n{:?}\n", top5_groundtruth, top5);
        for res in top5 {
            if top5_groundtruth.contains(res) {
                hit += 1;
            }
        }
        // println!("\ndeserialized_vamana_map {:?}\n\n", vamana_results.into_iter().map(|(_, i)|i).collect::<Vec<usize>>());
    }

    println!("5-recall-rate@5: {}", hit as f32 / (5 * round) as f32);

    // let serialized = serde_json::to_string(&vamana_map).unwrap();
    // let mut file = File::create("test_data/vamana_1m.json").unwrap();
    // file.write_all(serialized.as_bytes()).unwrap();

    // let mut file = File::open("test_data/vamana_1m.json").unwrap();
    // let mut contents = String::new();
    // file.read_to_string(&mut contents).unwrap();
    // let deserialized_vamana_map: FreshVamanaMap<Point, usize> =
    //     serde_json::from_str(&contents).expect("Failed to deserialize");
    // let vamana_results = deserialized_vamana_map.search(&Point(query.to_vec()));
    // println!("\ndeserialized_vamana_map {:?}\n\n", vamana_results);
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Point(Vec<f32>);

impl VamanaPoint for Point {
    // fn distance(&self, other: &Self) -> f32 {
    //     self.0
    //         .iter()
    //         .zip(other.0.iter())
    //         .map(|(a, b)| {
    //             let c = a - b;
    //             c * c
    //         })
    //         .sum::<f32>()
    //         .sqrt()
    // }

    fn distance(&self, other: &Self) -> f32 {
        assert_eq!(self.0.len(), other.0.len());

        let mut sum = f32x4::splat(0.0);
        let chunks = self.0.chunks_exact(4).zip(other.0.chunks_exact(4));

        for (a_chunk, b_chunk) in chunks {
            let a_simd = f32x4::from_slice(a_chunk);
            let b_simd = f32x4::from_slice(b_chunk);
            let diff = a_simd - b_simd;
            sum += diff * diff;
        }

        // Convert SIMD vector sum to an array and sum its elements
        let simd_sum: f32 = sum.to_array().iter().sum();

        // Handle remaining elements
        let remainder_start = self.0.len() - self.0.len() % 4;
        let remainder_sum: f32 = self.0[remainder_start..]
            .iter()
            .zip(&other.0[remainder_start..])
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum();

        // Calculate the total sum and then the square root
        (simd_sum + remainder_sum).sqrt()
    }

    fn dim() -> u32 {
        384
    }
    fn to_f32_vec(&self) -> Vec<f32> {
        self.0.iter().map(|v| *v).collect()
    }
    fn from_f32_vec(a: Vec<f32>) -> Self {
        Point(a.into_iter().collect())
    }
}
