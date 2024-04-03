use vectune::{Point as VamanaPoint, FreshVamanaMap, Builder as VamanaBuilder};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use serde::{Serialize, Deserialize};

use byteorder::{ReadBytesExt, LittleEndian};
use std::fs::File;
use std::io::{self, BufReader};
use std::io::{Read, Write};

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

  let mut rng = SmallRng::seed_from_u64(rand::random());

  let base_vectors = read_fvecs("./test_data/sift/sift_base.fvecs").unwrap();

  let mut points = Vec::new();
  let mut values = Vec::new();
  for (i, vec) in base_vectors.iter().enumerate() {
    points.push(Point(vec.to_vec()));
    values.push(i);
  }

  println!("building vamana...");
  let mut vamana_map: FreshVamanaMap<Point, usize> = VamanaBuilder::default().build(points.clone(), values.clone());
  vamana_map.ann.builder.l = 500;


  // Search in FreshVamana
  let query_vectors = read_fvecs("./test_data/sift/sift_query.fvecs").unwrap();
  let query_i = rng.gen_range(0..query_vectors.len() as usize);
  let query = &query_vectors[query_i];
  println!("searcing vamana...");
  let vamana_results = vamana_map.search(&Point(query.to_vec()));
  println!("{:?}\n\n", vamana_results);


  let groundtruth = read_ivecs("test_data/sift/sift_groundtruth.ivecs").unwrap();
  println!("groundtruth: {:?}", groundtruth[query_i]);
  println!("results: {:?}", vamana_results.iter().map(|(_, i)|*i).collect::<Vec<usize>>());



  let serialized = serde_json::to_string(&vamana_map).unwrap();
  let mut file = File::create("test_data/vamana_1m.json").unwrap();
  file.write_all(serialized.as_bytes()).unwrap();

  let mut file = File::open("test_data/vamana_1m.json").unwrap();
  let mut contents = String::new();
  file.read_to_string(&mut contents).unwrap();
  let deserialized_vamana_map: FreshVamanaMap<Point, usize> = serde_json::from_str(&contents).expect("Failed to deserialize");
  let vamana_results = deserialized_vamana_map.search(&Point(query.to_vec()));
  println!("\ndeserialized_vamana_map {:?}\n\n", vamana_results);

}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Point(Vec<f32>);

impl VamanaPoint for Point {
    fn distance(&self, other: &Self) -> f32 {
        self.0.iter()
          .zip(other.0.iter())
          .map(|(a, b)|{
            let c = a - b;
            c * c
          })
          .sum::<f32>()
          .sqrt()
    }

    fn dim() -> u32 {
      384
    }
    fn to_f32_vec(&self) -> Vec<f32> {
      self.0.iter().map(|v| {
        *v as f32
      }).collect()
    }
    fn from_f32_vec(a: Vec<f32>) -> Self {
      Point(a.into_iter().map(|v| v).collect())
    }
}