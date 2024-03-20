use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};


pub struct Builder {
  a: f32,
  r: usize,
  l: usize,
  seed: u64,
  // k: usize, // shards using k-means clustering,
  // l: usize, // the number of shards which the point belongs
}

impl Default for Builder {
  fn default() -> Self {
    Self {
      a: 1.2,
      r: 70,
      l: 125,
      seed: rand::random(),
    }
  }
}


impl Builder {

  pub fn set_l(&mut self, l: usize) {
    self.l = l;
  }

  pub fn build<T, P: Point, V: Clone>(self, points: Vec<P>, values: Vec<V>) -> FreshDiskAnnMap<P, V>{
    FreshDiskAnnMap::new(points, values, self)
  }
}



pub struct FreshDiskAnnMap<P, V> {
  ann: FreshDiskAnn<P>,
  values: Vec<V>,
}

impl<P, V> FreshDiskAnnMap<P, V>
where
    P: Point,
    V: Clone,
{
  fn new(points: Vec<P>, values: Vec<V>, builder: Builder) -> Self {
    let ann = FreshDiskAnn::new(points, builder);

    Self { ann, values}
  }
}

struct Node<P> {
  n_out: Vec<usize>, // has pointer. ToDo: should use PQ to resuce memory accesses.
  n_in: Vec<usize>,
  p: P,
  id: usize,
}

struct FreshDiskAnn<P>
{
  nodes: Vec<Node<P>>,
  centroid: usize,
}

impl<P> FreshDiskAnn<P>
where
    P: Point,
{
  pub fn new(points: Vec<P>, builder: Builder) -> Self {
    // Initialize Random Graph
    let ann = FreshDiskAnn::<P>::random_graph_init(points, builder);

    // Robust Prune

    ann
  }

  fn random_graph_init(points: Vec<P>, builder: Builder) -> Self {
    // let a = builder.a;
    let r = builder.r;
    // let l = builder.l;
    let mut rng = SmallRng::seed_from_u64(builder.seed);

    if points.is_empty() {
      return Self {
          nodes: Vec::new(),
          centroid: usize::MAX,
        }
    }

    assert!(points.len() < u32::MAX as usize);
    let points_len = points.len();

    // Find Centroid
    let mut average_point: Vec<f32> = vec![0.0; P::dim() as usize];
    for p in &points {
      average_point = p.to_f32_vec().iter().zip(average_point.iter()).map(|(x, y)| x + y).collect();
    }
    let average_point = P::from_f32_vec(average_point.into_iter().map(|v| v / points_len as f32).collect());
    let mut min_dist = f32::MAX;
    let mut centroid = usize::MAX;
    for (i, p) in points.iter().enumerate() {
      let dist = p.distance(&average_point);
      if dist < min_dist {
        min_dist = dist;
        centroid = i;
      }
    }


    // Get random connected graph
    let mut nodes: Vec<Node<P>> = points.into_iter().enumerate().map(|(id, p)| Node {
      n_out: Vec::new(),
      n_in: Vec::new(),
      p,
      id
    }).collect();

    for self_i in 0..points_len {
      let self_node = &mut nodes[self_i];
      // Add random out nodes
      let mut back_links = Vec::new();
      while self_node.n_out.len() < r {
        let out_i = rng.gen_range(0..points_len as usize);

        // To not contain self-reference and duplication
        if out_i == self_i || self_node.n_out.contains(&out_i) {
          continue;
        }

        self_node.n_out.push(out_i);
        back_links.push(out_i);
      }

      for out_i in back_links {
        let out_node = &mut nodes[out_i];
        out_node.n_in.push(self_i);
      }
    }

    Self {
      nodes,
      centroid,
    }

  }

  fn greedy_search(&self, xq: P, k: usize, l: usize) -> (Vec<(f32, usize)>, Vec<usize>) { // k-anns, visited
    assert!(l >= k);
    let s = self.centroid;
    let mut visited: Vec<usize> = Vec::new();
    let mut list: Vec<(f32, usize)> = vec![(self.nodes[s].p.distance(&xq), s)];

    fn set_diff(a: Vec<(f32, usize)>, b: &Vec<usize>) -> Vec<(f32, usize)> {
      a.into_iter().filter(|(_, p)| !b.contains(p)).collect()
    }

    fn find_nearest(c: &mut Vec<(f32, usize)>) -> usize {
      c.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less));
      c[0].1
    }

    fn resize(list: &mut Vec<(f32, usize)>, size: usize) {
      list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less));
      list.truncate(size)
    }

    let mut working = list.clone(); // Because list\visited == list at beginning
    while working.len() > 0 {

      println!("list: {:?}, visited: {:?} \n\n\n", list, visited);


      let nearest = find_nearest(&mut working);

      if visited.contains(&nearest) {
        continue;
      } else {
        visited.push(nearest)
      }

      for out_i in &self.nodes[nearest].n_out {
        let node = &self.nodes[*out_i];
        let node_i = node.id;

        let is_contained_in_list = list.iter().filter(|(_, id)| *id == node_i).collect::<Vec<&(f32, usize)>>().len() != 0;
        if is_contained_in_list {
          continue;
        }

        let dist = xq.distance(&node.p);
        list.push((dist, node_i));
      }

      if list.len() > l {
        resize(&mut list, l)
      }

      working = set_diff(list.clone(), &visited);
    }

    resize(&mut list, k);
    let k_anns = list;

    (k_anns, visited)

  }

  fn robust_prune(&mut self, point_id: usize) {
    
  }
}



pub trait Point: Clone + Sync {
    fn distance(&self, other: &Self) -> f32;
    fn dim() -> u32;
    fn to_f32_vec(&self) -> Vec<f32>;
    fn from_f32_vec(a: Vec<f32>) -> Self;
}



#[cfg(test)]
mod tests {
  use super::*;
  use rand::rngs::SmallRng;
  use rand::{Rng, SeedableRng};


  #[derive(Clone, Debug)]
  struct Point(Vec<u32>);
  impl crate::Point for Point {
      fn distance(&self, other: &Self) -> f32 {
          self.0.iter()
            .zip(other.0.iter())
            .map(|(a, b)| (*a as f32 - *b as f32).powi(2))
            .sum::<f32>()
            .sqrt()
      }
      fn dim() -> u32 {
        3
      }
      fn to_f32_vec(&self) -> Vec<f32> {
        self.0.iter().map(|v| {
          *v as f32
        }).collect()
      }
      fn from_f32_vec(a: Vec<f32>) -> Self {
        Point(a.into_iter().map(|v| v as u32).collect())
      }
  }

  #[test]
  fn fresh_disk_ann_new_empty() {
    let builder = Builder::default();

    let ann: FreshDiskAnn<Point> = FreshDiskAnn::random_graph_init(Vec::new(), builder);
    assert_eq!(ann.nodes.len(), 0);
  }

  #[test]
  fn fresh_disk_ann_new_r() {

    let builder = Builder::default();
    let mut rng = SmallRng::seed_from_u64(builder.seed);
    let r = builder.r;

    let points: Vec<Point> = (0..100).into_iter().map(|_| {
      let a = rng.gen::<u32>();
      Point(vec![a;3])
    }).collect();

    let ann: FreshDiskAnn<Point> = FreshDiskAnn::random_graph_init(points, builder);
    for node in ann.nodes {
      assert_eq!(node.n_out.len(), r);
      assert_ne!(node.n_in.len(), 0);
    }
  }

  #[test]
  fn fresh_disk_ann_new_centroid() {

    let builder = Builder::default();

    let mut i = 0;

    let points: Vec<Point> = (0..100).into_iter().map(|_| {
      let a = i;
      i += 1;
      Point(vec![a;3])
    }).collect();

    let ann: FreshDiskAnn<Point> = FreshDiskAnn::random_graph_init(points, builder);
    assert_eq!(ann.centroid, 49);
  }

  #[test]
  fn greedy_search() {

    let mut builder = Builder::default();
    builder.set_l(30);
    let l = builder.l;

    let mut i = 0;

    let points: Vec<Point> = (0..100).into_iter().map(|_| {
      let a = i;
      i += 1;
      Point(vec![a;3])
    }).collect();

    let ann: FreshDiskAnn<Point> = FreshDiskAnn::random_graph_init(points, builder);
    let xq = Point(vec![0;3]);
    let k = 10;
    let (k_anns, _visited) = ann.greedy_search(xq, k, l);

    for i in 0..10 {
      assert_eq!(k_anns[i].1, i);
    }
  }
}
