use serde::Deserialize;
use std::fs;
use vectune::{Builder as VamanaBuilder, FreshVamanaMap, Point as VamanaPoint};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

#[derive(Deserialize, Debug)]
// #[serde(deny_unknown_fields)]
struct Data {
    sentence: String,
    embeddings: Vec<f32>,
}

fn main() {
    #[cfg(feature = "nightly")]
    println!("nightly simd mode");
    #[cfg(not(feature = "nightly"))]
    println!("normal mode");

    let mut rng = SmallRng::seed_from_u64(rand::random());
    let file_path = "./embeddings_wiki_random_half.json";

    let data = fs::read_to_string(file_path).expect("Failed to read file");

    let embeddings_data: Vec<Data> = serde_json::from_str(&data).expect("Failed to deserialize");

    let mut points = Vec::new();
    let mut values = Vec::new();
    for d in embeddings_data {
        points.push(Point(d.embeddings));
        values.push(d.sentence);
    }

    // Random query
    let query_point = &points[rng.gen_range(0..points.len())];

    println!("building vamana...");
    let vamana_map: FreshVamanaMap<Point, String> =
        VamanaBuilder::default().build(points.clone(), values.clone());

    // Search in FreshVamana
    println!("searcing vamana...");
    let vamana_results = vamana_map.search(query_point);
    println!("{:?}\n\n", vamana_results);
}

#[derive(Clone, Debug)]
struct Point(Vec<f32>);

impl VamanaPoint for Point {
    fn distance(&self, other: &Self) -> f32 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| {
                let c = a - b;
                c * c
            })
            .sum::<f32>()
        // .sqrt()
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
