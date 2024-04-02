// use std::collections::HashMap;
// use hashbrown::HashMap;

use ahash::AHasher;
// use std::hash::{BuildHasherDefault, Hash};
use std::collections::HashMap;
use std::time::Instant;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand::seq::SliceRandom;

use itertools::Itertools;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

pub mod pq;
pub mod kmeans;

use kmeans::KMeans;

/*
ToDo:
  - Testing PQ
  - Modify insert to mach PQ

  - PQを使ったGreedySearchで、Noutの検索にはPQを使って、RobustPruneではoriginalを使うように変える。
*/

/*
Debug Note:
  - After indexing three times, the graph is quite healthy. Figure out why it is.
*/

/*
Note API
  - Prepare insert and union separately as api.　
    The insert allows duplicates, while the union does not insert if the file already exists.

  このリポジトリは、VamanaIndexingの高速buildで行うにとどめて、検索は別のlibに譲る。
  量子化やそれと使った検索はグラフビルドとは独立して行うように。
  グラフビルド時には量子化は使わない。精度の低下に比べてビルド速度がよくならない可能性。
  その代わり、並列化とSIMDを多用できるよう設計し直す。
  グラフのデータ構造は、なるべく簡素に。
*/

#[derive(Serialize, Deserialize, Clone)]
pub struct Builder {
  a: f32,
  r: usize,
  pub l: usize,
  seed: u64,
  // k: usize, // shards using k-means clustering,
  // l: usize, // the number of shards which the point belongs

  pq_m: usize,
  pq_k: usize,
}

impl Default for Builder {
  fn default() -> Self {
    Self {
      a: 2.0,
      r: 70,
      l: 125,
      seed: rand::random(),

      //PQ
      pq_m: 16,
      pq_k: 256,
    }
  }
}


impl Builder {
  pub fn set_a(&mut self, a: f32) {
    self.a = a;
  }
  pub fn set_r(&mut self, r: usize) {
    self.r = r;
  }
  pub fn set_l(&mut self, l: usize) {
    self.l = l;
  }
  pub fn set_seed(&mut self, seed: u64) {
    self.seed = seed;
  }
  pub fn set_pq_m(&mut self, pq_m: usize) {
    self.pq_m = pq_m;
  }
  pub fn set_pq_k(&mut self, pq_k: usize) {
    self.pq_k = pq_k;
  }
  
  pub fn build<P: Point, V: Clone>(self, points: Vec<P>, values: Vec<V>) -> FreshVamanaMap<P, V>{
    FreshVamanaMap::new(points, values, self)
  }
}


#[derive(Serialize, Deserialize)]
pub struct FreshVamanaMap<P, V> {
  pub ann: FreshVamana<P>,
  values: Vec<V>,
}

impl<P, V> FreshVamanaMap<P, V>
where
    P: Point,
    V: Clone,
{
  fn new(points: Vec<P>, values: Vec<V>, builder: Builder) -> Self {
    let ann = FreshVamana::new(points, builder);

    Self {ann, values}
  }
  pub fn search(&self, query_point: &P) -> Vec<(f32, V)> {
    let (results, _visited) = self.ann.greedy_search(&query_point, 30, self.ann.builder.l);
    results.into_iter().map(|(dist, i)| (dist, self.values[i].clone())).collect()
  }
}

#[derive(Serialize, Deserialize, Clone)]
struct Node<P> {
  n_out: Vec<usize>, // has pointer. ToDo: should use PQ to resuce memory accesses.
  n_in: Vec<usize>,
  p: P,
  id: usize,
  pq: Vec<usize>,
}

impl<P> Node<P> {
  fn new(p: P, pq: Vec<usize>, id: usize) -> Self {
    Self {
      n_out: Vec::new(),
      n_in: Vec::new(),
      p,
      id,
      pq
    }
  }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct FreshVamana<P>
{
  nodes: Vec<Node<P>>,
  centroid: usize,
  pub builder: Builder,
  cemetery: Vec<usize>,
  empties: Vec<usize>,
  dist_cache: HashMap<(usize, usize), f32>,
  codebooks: Vec<Vec<P>>,
  pq_dist_map: HashMap<(usize, usize, usize), f32>,
}

impl<P> FreshVamana<P>
where
    P: Point,
{
  pub fn new(points: Vec<P>, builder: Builder) -> Self {
    let mut rng = SmallRng::seed_from_u64(builder.seed);
    println!("seed: {}", builder.seed);

    println!("quantizing phase");
    let start_time = Instant::now();


    /*
      Note: 
      Disable the quantization for now
    */
    // let (points, codebooks) = PQ::new(rng.clone(), builder.pq_m, builder.pq_k, P::dim() as usize, points).quantize();
    let codebooks = vec![];
    let points = points.into_iter().map(|p| (p, vec![])).collect();

    println!("\nquantizing time: {:?}", Instant::now().duration_since(start_time));

    println!("pq dist phase");
    let start_time = Instant::now();
    let pq_dist_map = HashMap::new();
    println!("\npq dist time: {:?}", Instant::now().duration_since(start_time));


    // Initialize Random Graph
    println!("rand init phase");
    let start_time = Instant::now();
    let mut ann = FreshVamana::<P>::random_graph_init_v3(points, builder, &mut rng, codebooks, pq_dist_map);
    println!("\nrand init time: {:?}", Instant::now().duration_since(start_time));

    // Prune Edges

    // let σ denote a random permutation of 1..n
    let node_len = ann.nodes.len();
    let mut shuffled: Vec<(usize, usize)> = (0..node_len).into_iter().map(|node_i| (rng.gen_range(0..node_len as usize), node_i)).collect();
    shuffled.sort_by(|a, b| a.0.cmp(&b.0));

    FreshVamana::<P>::para_indexing(&mut ann, shuffled);


    // /*
    // Note:
    //   It is important to index twice from a randomly initialized graph.
    //   One-time indexing would create unreferenced nodes from the neighborhood.
    //   Since nodes which selected relatively at the beginning don't have actual neighborhoods in its n_out.
    // */
    // println!("\n\n\n");
    // let mut shuffled: Vec<(usize, usize)> = (0..node_len).into_iter().map(|node_i| (rng.gen_range(0..node_len as usize), node_i)).collect();
    // shuffled.sort_by(|a, b| a.0.cmp(&b.0));

    // FreshVamana::<P>::indexing(&mut ann, shuffled);

    println!("dist cache len: {} / {}", ann.dist_cache.len(), (ann.nodes.len() * ann.nodes.len()) / 2);

    println!("\ntotal indexing time: {:?}", Instant::now().duration_since(start_time));



    ann
  }

  fn random_graph_init_v3(points: Vec<(P, Vec<usize>)>, builder: Builder, rng: &mut SmallRng, codebooks: Vec<Vec<P>>, pq_dist_map: HashMap<(usize, usize, usize), f32>) -> Self {

    if points.is_empty() {
      return Self {
          nodes: Vec::new(),
          centroid: usize::MAX,
          builder,
          cemetery: Vec::new(),
          empties: Vec::new(),
          dist_cache: HashMap::new(),
          codebooks: Vec::new(),
          pq_dist_map: HashMap::new(),
        }
    }

    assert!(points.len() < u32::MAX as usize);
    let points_len = points.len();

    /* Find Centroid */
    let mut average_point: Vec<f32> = vec![0.0; P::dim() as usize];
    for p in &points {
      average_point = p.0.to_f32_vec().iter().zip(average_point.iter()).map(|(x, y)| x + y).collect();
    }
    let average_point = P::from_f32_vec(average_point.into_iter().map(|v| v / points_len as f32).collect());
    let mut min_dist = f32::MAX;
    let mut centroid = usize::MAX;
    for (i, p) in points.iter().enumerate() {
      let dist = p.0.distance(&average_point);
      if dist < min_dist {
        min_dist = dist;
        centroid = i;
      }
    }


    /* Get random connected graph */
    let mut nodes: Vec<Node<P>> = points.clone().into_iter().enumerate().map(|(id, (p, pq))| Node {
      n_out: Vec::new(),
      n_in: Vec::new(),
      p,
      id,
      pq
    }).collect();

    let node_len = nodes.len();
    let r_size = builder.r;

    /* Clustering */
    let mut shard_num = 2;
    if points_len > 50_000 {
      shard_num += points_len / 50_000;
    }

    println!("shard_num: {}", shard_num);

    let mut kmeans: KMeans<P> = KMeans::new(rng.clone(), shard_num, 100, points.into_iter().map(|(p, _)| p).collect());
    println!(" kmeans.kmeans_pp_centroids()...");
    let initial_centorids = kmeans.kmeans_pp_centroids();
    println!(" kmeans.calculate(initial_centorids)...");
    let (shards, _) = kmeans.calculate(initial_centorids);
    // println!("kmeans.calculate shard len: {:?}", shards);
    println!(" making shards...");
    let mut shards: Vec<(usize, usize)> = shards
      .into_iter()
      .enumerate()
      .map(|(node_i, cluster_id)| (cluster_id, node_i))
      .collect();
    shards.sort();
    // for shard in &shards {
    //   println!("each shard len: {:?}", shard);
    // }
    println!("shuffled shard len: {}", shards.len());

    let shards: Vec<Vec<usize>> = shards
      .into_iter()
      .group_by(|&(first, _)| first)
      .into_iter()
      .map(|(_key, group)| group.map(|(_, i)| i).collect())
      .collect();

    for shard in &shards {
      println!("each shard len: {}", shard.len());
    }

    /* Random initialization with cluster priority */
    println!(" Random initialization...");
    let shuffled_node_ids = (0..node_len).collect::<Vec<_>>();
    let mut shuffled_node_ids: Vec<usize> = (0..(r_size/3)+1).into_iter().map(|_| shuffled_node_ids.clone()).flatten().collect();
    shuffled_node_ids.shuffle(rng);

    println!(" Iter shards...");
    for (shard_i, shard) in shards.iter().enumerate() {
      let start_idx = shard_i * r_size/3;
      let end_idx = start_idx + r_size/3;
      let other_shard_nodes = &shuffled_node_ids[start_idx..end_idx];
      
      let mut candidates: Vec<usize> = (0..(2*r_size/3)+1).into_iter().map(|_| shard.clone()).flatten().collect();
      candidates.extend(other_shard_nodes.iter()); // ここを直す？
      candidates.shuffle(rng);

      let shard_len = shard.len();

      println!("shard len: {}, candidates len: {}, (0..(2*r_size/3)+1) len: {}", shard_len, candidates.len(), (0..(2*r_size/3)+1).len());

      for (i, node_i) in shard.into_iter().enumerate() {
        let start = i*r_size % candidates.len();
        // println!("start idx {}", start);
        let end = if start + r_size > candidates.len() {
          candidates.len()
        } else {
          start + r_size
        };
        let mut new_n_out = candidates[start..end].to_vec();
        new_n_out.sort();
        new_n_out.dedup();
  
        for out_i in &new_n_out {
          insert_id(*node_i, &mut nodes[*out_i].n_in);
        }
  
        nodes[*node_i].n_out = new_n_out;
      }

      // for rest_i in candidates[]
      // wip 残ったやつを割り振る。
    }

    let node_len = nodes.len();
    Self {
      nodes,
      centroid,
      builder,
      cemetery: Vec::new(),
      empties: Vec::new(),
      dist_cache: HashMap::with_capacity(node_len/2),
      codebooks,
      pq_dist_map,
    }

  }

  fn random_graph_init_v2(points: Vec<(P, Vec<usize>)>, builder: Builder, rng: &mut SmallRng, codebooks: Vec<Vec<P>>, pq_dist_map: HashMap<(usize, usize, usize), f32>) -> Self {

    if points.is_empty() {
      return Self {
          nodes: Vec::new(),
          centroid: usize::MAX,
          builder,
          cemetery: Vec::new(),
          empties: Vec::new(),
          dist_cache: HashMap::new(),
          codebooks: Vec::new(),
          pq_dist_map: HashMap::new(),
        }
    }

    assert!(points.len() < u32::MAX as usize);
    let points_len = points.len();

    let mut shard_num = 1;

    if points_len > 100_000 {
      shard_num += (points_len - 1) / 100_000;
    }

    /* Find Centroid */
    let mut average_point: Vec<f32> = vec![0.0; P::dim() as usize];
    for p in &points {
      average_point = p.0.to_f32_vec().iter().zip(average_point.iter()).map(|(x, y)| x + y).collect();
    }
    let average_point = P::from_f32_vec(average_point.into_iter().map(|v| v / points_len as f32).collect());
    let mut min_dist = f32::MAX;
    let mut centroid = usize::MAX;
    for (i, p) in points.iter().enumerate() {
      let dist = p.0.distance(&average_point);
      if dist < min_dist {
        min_dist = dist;
        centroid = i;
      }
    }


    /* Get random connected graph */
    let mut nodes: Vec<Node<P>> = points.into_iter().enumerate().map(|(id, (p, pq))| Node {
      n_out: Vec::new(),
      n_in: Vec::new(),
      p,
      id,
      pq
    }).collect();
    
    let node_len = nodes.len();

    let r_size = builder.r;

    let mut shuffle_node_ids = (0..node_len).collect::<Vec<_>>();

    let shuffle_node_ids_duplicated_r_times: Vec<usize> = (0..r_size).into_iter().map(|_| {
      shuffle_node_ids.shuffle(rng);
      shuffle_node_ids.clone()
    }).flatten().collect();

    shuffle_node_ids.shuffle(rng);

    for (i, node_i) in shuffle_node_ids.into_iter().enumerate() {
      let mut new_n_out = shuffle_node_ids_duplicated_r_times[i..i+r_size].to_vec();
      new_n_out.sort();
      new_n_out.dedup();

      for out_i in &new_n_out {
        insert_id(node_i, &mut nodes[*out_i].n_in);
      }

      nodes[node_i].n_out = new_n_out;

    }

    // for (id, node) in nodes.iter().enumerate() {
    //   println!("id: {}, len: {}, n_out: {:?}", id, node.n_out.len(), node.n_out);
    // }

    let node_len = nodes.len();
    Self {
      nodes,
      centroid,
      builder,
      cemetery: Vec::new(),
      empties: Vec::new(),
      dist_cache: HashMap::with_capacity(node_len/2),
      codebooks,
      pq_dist_map,
    }

  }


  fn random_graph_init(points: Vec<(P, Vec<usize>)>, builder: Builder, rng: &mut SmallRng, codebooks: Vec<Vec<P>>, pq_dist_map: HashMap<(usize, usize, usize), f32>) -> Self {

    if points.is_empty() {
      return Self {
          nodes: Vec::new(),
          centroid: usize::MAX,
          builder,
          cemetery: Vec::new(),
          empties: Vec::new(),
          dist_cache: HashMap::new(),
          codebooks: Vec::new(),
          pq_dist_map: HashMap::new(),
        }
    }

    assert!(points.len() < u32::MAX as usize);
    let points_len = points.len();

    /* Find Centroid */
    let mut average_point: Vec<f32> = vec![0.0; P::dim() as usize];
    for p in &points {
      average_point = p.0.to_f32_vec().iter().zip(average_point.iter()).map(|(x, y)| x + y).collect();
    }
    let average_point = P::from_f32_vec(average_point.into_iter().map(|v| v / points_len as f32).collect());
    let mut min_dist = f32::MAX;
    let mut centroid = usize::MAX;
    for (i, p) in points.iter().enumerate() {
      let dist = p.0.distance(&average_point);
      if dist < min_dist {
        min_dist = dist;
        centroid = i;
      }
    }


    /* Get random connected graph */
    let mut nodes: Vec<Node<P>> = points.into_iter().enumerate().map(|(id, (p, pq))| Node {
      n_out: Vec::new(),
      n_in: Vec::new(),
      p,
      id,
      pq
    }).collect();
    
    let mut working: Vec<usize> = (0..nodes.len()).collect();
    let node_len = nodes.len();

    for node_i in 0..node_len {
      let mut n_out_cout = 0;
      while n_out_cout < builder.r {
        let working_i = rng.gen_range(0..working.len() as usize);
        if working_i == node_i {
          continue;
        } else {
          n_out_cout += 1;
        }
        let out_node_i = working[working_i];
        insert_id(node_i, &mut nodes[out_node_i].n_in); // ToDo: refactor , use self.make_edge()
        insert_id(out_node_i, &mut nodes[node_i].n_out);

        // Since prevents the creation of nodes that are not referenced by anyone during initialization,
        // ensure that all input edges are R nodes
        if nodes[out_node_i].n_in.len() == builder.r {
          working.remove(working_i);
        }
      }
    }

    let node_len = nodes.len();
    Self {
      nodes,
      centroid,
      builder,
      cemetery: Vec::new(),
      empties: Vec::new(),
      dist_cache: HashMap::with_capacity(node_len/2),
      codebooks,
      pq_dist_map,
    }

  }


  fn indexing(ann: &mut FreshVamana<P>, shuffled: Vec<(usize, usize)>) {

    // for 1 ≤ i ≤ n do
    for (count, (_, i)) in shuffled.into_iter().enumerate() {

      println!("id : {}\t/{}", count, ann.nodes.len());

      // let [L; V] ← GreedySearch(s, xσ(i), 1, L)
      let (_, visited) = ann.greedy_search(&ann.nodes[i].p, 1, ann.builder.l);
      // run RobustPrune(σ(i), V, α, R) to update out-neighbors of σ(i)
      ann.robust_prune(i, visited);

      // for all points j in Nout(σ(i)) do
      for j in ann.nodes[i].n_out.clone() {
        if ann.nodes[j].n_out.contains(&i) {
          continue;
        } else {
          // Todo : refactor, self.make_edge　or union. above ann.nodes[j].n_out.contains(&i) not necessary if use union
          insert_id(i, &mut ann.nodes[j].n_out);
          insert_id(j, &mut ann.nodes[i].n_in);
        }

        // if |Nout(j) ∪ {σ(i)}| > R then run RobustPrune(j, Nout(j) ∪ {σ(i)}, α, R) to update out-neighbors of j
        if ann.nodes[j].n_out.len() > ann.builder.r {
          // robust_prune requires (dist(xp, p'), index)
          let v: Vec<(f32, usize)> = ann.nodes[j].n_out.clone().into_iter()
            .map(|out_i: usize| 
              // (ann.nodes[out_i].p.distance(j_point), out_i)
              (ann.node_distance(out_i, j), out_i)
            ).collect();

          ann.robust_prune(j, v);
        }
      }
    }
  }

  fn para_indexing(ann: &mut FreshVamana<P>, shuffled: Vec<(usize, usize)>) {
    /*
    Note:
      n_inは最後に追加すればいいので、並列ループないでは追加しない。依存性をなくすため。
      Noutを最終的にマージするように。
    */


    let ann_fix = ann.clone();

    let start_time =  Instant::now();

    let mut all_edges: Vec<(usize, usize)> = shuffled.into_par_iter().enumerate().map(|(count, (_, node_i))| {

      let (_, visited) = ann_fix.greedy_search(&ann_fix.nodes[node_i].p, 1, ann_fix.builder.l);

      if count % 10000 == 0 {
        println!("visiting phase, id : {}\t/{} visited len: {} passed time: {:?}", count, ann.nodes.len(), visited.len(), Instant::now().duration_since(start_time));
      }

      // Joint visited ids and current n_out ids
      let mut out_ids: Vec<usize> = visited.into_iter().map(|(_, id)| id).collect(); // ToDo: reuse dist
      out_ids.extend(ann_fix.nodes[node_i].n_out.iter().map(|out_i| *out_i));
      out_ids.sort();
      out_ids.dedup();
      

      // Make edges: all out_i ids has backlinks.
      let mut edges: Vec<(usize, usize)> = out_ids.clone().into_iter().map(|out_i| (node_i, out_i)).collect();
      edges.extend(out_ids.into_iter().map(|from_i| (from_i, node_i)));

      edges
    }).flatten().collect();
    println!("\nvisiting time: {:?}", Instant::now().duration_since(start_time));


    let start_time = Instant::now();
    all_edges.sort();
    all_edges.dedup();
    let groups: Vec<(usize, Vec<usize>)> = all_edges
      .into_iter()
      .group_by(|&(first, _)| first)
      .into_iter()
      .map(|(key, group)| (key, group.map(|(_, i)| i).collect()))
      .collect();

    println!("\ngrouping time: {:?}", Instant::now().duration_since(start_time));

    let start_time =  Instant::now();
    let pruned_edges: Vec<(usize, Vec<usize>)> = groups.into_par_iter().map(|(node_i, n_out)| {

      if node_i % 10000 == 0 {
        println!("robust pruning phase, id : {}\t/{}, passed time: {:?}", node_i, ann.nodes.len(), Instant::now().duration_since(start_time));
      }

      let mut candidates: Vec<(f32, usize)> = n_out.into_iter().map(|out_i| {
        let node_i_point =  &ann.nodes[node_i].p;
        let out_i_point = &ann.nodes[out_i].p;
        let dist = node_i_point.distance(out_i_point);
        (dist, out_i)
      }).collect();
      sort_list_by_dist(&mut candidates);

      // V ← (V ∪ Nout(p)) \ {p}
      remove_from(&(0.0, node_i), &mut candidates);

      let mut n_out = vec![];

      while let Some((first, rest)) = candidates.split_first() {
        let (_, pa) = first.clone(); // pa is p asterisk (p*), which is nearest point to p in this loop
        n_out.push(pa);

        if n_out.len() == ann.builder.r {
          break;
        }
        candidates = rest.to_vec();

        // if α · d(p*, p') <= d(p, p') then remove p' from v
        candidates.retain(|&(dist_xp_pd, pd)| {
          let pa_point =  &ann.nodes[pa].p;
          let pd_point = &ann.nodes[pd].p;
          let dist_pa_pd = pa_point.distance(pd_point);

          ann.builder.a * dist_pa_pd > dist_xp_pd
        })
      }

      (node_i, n_out)

    }).collect();

    for (from_i, to_ids) in pruned_edges {
      for to_i in to_ids {
        insert_id(to_i, &mut ann.nodes[from_i].n_out);
        insert_id(from_i, &mut ann.nodes[to_i].n_in);
      }
    }

    println!("\nrobust pruning time: {:?}", Instant::now().duration_since(start_time));

  }

  fn para_indexing_v1(ann: &mut FreshVamana<P>, shuffled: Vec<(usize, usize)>) {
    /*
    Note:
      n_inは最後に追加すればいいので、並列ループないでは追加しない。依存性をなくすため。
      Noutを最終的にマージするように。
    */


    let ann_fix = ann.clone();

    let start_time =  Instant::now();

    for (count, (_, i)) in shuffled.iter().enumerate() {


      let i = *i;
      let (_, visited) = ann_fix.greedy_search(&ann_fix.nodes[i].p, 1, ann_fix.builder.l);
      // let (_, visited) = ann_fix.greedy_search_pq(i, 1, ann_fix.builder.l);

      if count % 5000 == 0 {
        println!("visiting phase, id : {}\t/{} visited len: {} passed time: {:?}", count, ann.nodes.len(), visited.len(), Instant::now().duration_since(start_time));
      }

      let mut visited_ids: Vec<usize> = visited.into_iter().map(|(_, id)| id).collect(); // ToDo: reuse dist
      visited_ids.sort();

      // ann.nodes[i].n_outを上書きしないように気を付ける。
      for id in visited_ids {
        insert_id(id, &mut ann.nodes[i].n_out);
        insert_id(i,  &mut ann.nodes[id].n_in);
      }

      for j in ann.nodes[i].n_out.clone() {

        insert_id(i, &mut ann.nodes[j].n_out);
        insert_id(j, &mut ann.nodes[i].n_in);

      }

      /*
      Note:
      Noutとvisitedを結合する。
      それらを相互リンクする。
      
      */
    }

    println!("\nvisiting time: {:?}", Instant::now().duration_since(start_time));

    let start_time =  Instant::now();

    // メモ： N_outがソートされてないで挿入されている。union_idがおかしい？ insertを使うべき？


    // Noutを並列でRobustPruneできるようにする。
    for (count, (_, i)) in shuffled.iter().enumerate() {

      if count % 5000 == 0 {
        println!("robust pruning phase, id : {}\t/{}, passed time: {:?}", count, ann.nodes.len(), Instant::now().duration_since(start_time));
      }
      
      let mut v: Vec<(f32, usize)> = ann.nodes[*i].n_out.clone().into_iter().map(|j| {
        (ann.node_distance(*i, j), j)
      }).collect();

      sort_list_by_dist(&mut v);

      // println!("id: {}, v {:?}\n\n", i, v);

      ann.robust_prune(*i, v);
    }

    println!("\nrobust pruning time: {:?}", Instant::now().duration_since(start_time));

  }

  fn node_distance(&mut self, a: usize, b: usize) -> f32 {
    self.nodes[a].p.distance(&self.nodes[b].p)
  }

  // fn node_distance(&mut self, a: usize, b: usize) -> f32 {
  //   let key = (std::cmp::min(a, b), std::cmp::max(a, b));
  //   match self.dist_cache.get(&key) {
  //     Some(dist) => {
  //       // println!("cache is used {:?}", key);
  //       *dist
  //     },
  //     None => {
  //       let dist = self.nodes[a].p.distance(&self.nodes[b].p);
  //       self.dist_cache.insert(key, dist);
  //       dist
  //     },
  //   }
  // }

  pub fn insert(&mut self, p: P, pq: Vec<usize>) {
    // Add node

    let pid =  if self.empties.len() == 0 { // ToDo: cache
      let id = self.nodes.len();
      self.nodes.push(Node::new(p.clone(), pq.clone(), id));
      id
    } else {
      let id = self.empties[0];
      self.empties.remove(0);
      self.nodes[id] = Node::new(p.clone(), pq.clone(),id);
      id
    };


    // [L, V] ← GreedySearch(𝑠, 𝑝, 1, 𝐿)
    let (_list, visited) = self.greedy_search(&p, 1, self.builder.l);
    // 𝑁out(𝑝) ← RobustPrune(𝑝, V, 𝛼, 𝑅) (Algorithm 3)
    self.robust_prune(pid, visited);
    // foreach 𝑗 ∈ 𝑁out(𝑝) do
    for j in self.nodes[pid].n_out.clone() {
      // |𝑁out(𝑗) ∪ {𝑝}| 
      insert_id(pid, &mut self.nodes[j].n_out);
      // if |𝑁out(𝑗) ∪ {𝑝}| > 𝑅 then 
      let j_n_out = self.nodes[j].n_out.clone();
      let j_point = self.nodes[j].p.clone();
      if j_n_out.len() > self.builder.r {
        // 𝑁out(𝑗) ← RobustPrune(𝑗, 𝑁out(𝑗) ∪ {𝑝}, 𝛼, 𝑅)
        let mut j_n_out_with_dist = j_n_out.into_iter().map(|id| (j_point.distance(&self.nodes[id].p), id)).collect::<Vec<(f32, usize)>>();
        sort_list_by_dist(&mut j_n_out_with_dist);
        self.robust_prune(j, j_n_out_with_dist);
      }
    }

  }

  pub fn inter(&mut self, node_i: usize) {
    if !self.cemetery.contains(&node_i) {
      insert_id(node_i, &mut self.cemetery)
    }
  }

  // fn make_edge(out_i: usize, in_i: usize) { // out_i -> in_i
  //   // todo make sure adding backlinks
  //   todo!();
  // }

  pub fn remove_graves(&mut self) {


    /* 𝑝 ∈ 𝑃 \ 𝐿𝐷 s.t. 𝑁out(𝑝) ∩ 𝐿𝐷 ≠ ∅ */

    // Note: 𝐿𝐷 is Deleted List
    let mut ps = Vec::new();
    // s.t. 𝑁out(𝑝) ∩ 𝐿𝐷 ≠ ∅
    for grave_i in self.cemetery.clone() {
      // println!("grave id {}, out: {:?}", grave_i, self.nodes[*grave_i].n_in);
      ps = union_ids(&ps, &self.nodes[grave_i].n_in);
      // println!("{}: grave_n_in: {:?}, ps: {:?}" , grave_i, self.nodes[grave_i].n_in, ps);
    }
    // 𝑝 ∈ 𝑃 \ 𝐿𝐷
    ps = diff_ids(&ps, &self.cemetery);

    // println!("self.cemetery {:?} ", self.cemetery);

    for p in ps {
      // D ← 𝑁out(𝑝) ∩ 𝐿𝐷
      let d = intersect_ids(&self.nodes[p].n_out, &self.cemetery);
      // C ← 𝑁out(𝑝) \ D //initialize candidate list
      let mut c = diff_ids(&self.nodes[p].n_out, &d);
      // println!("id: {}, D {:?} , C: {:?}", p, d, c);

      // foreach 𝑣 ∈ D do
      for u in &d {
        // C ← C ∪ 𝑁out(𝑣)
        c = union_ids(&c, &self.nodes[*u].n_out);
      }

      // C ← C \ D
      /*
       Note:
        Since D's Nout may contain LD, Why pull the D instead of the LD?
        I implemented it as shown and it hit data that should have been erased, so I'll fix it to pull LD.
        `c = diff_ids(&c, &d); // <- ???`
      */
      c = diff_ids(&c, &self.cemetery);

      // 𝑁out(𝑝) ← RobustPrune(𝑝, C, 𝛼, 𝑅)
      let p_point = self.nodes[p].p.clone();
      let mut c_with_dist: Vec<(f32, usize)> = c.into_iter().map(|id| (p_point.distance(&self.nodes[id].p), id)).collect();

      sort_list_by_dist(&mut c_with_dist);
      /* 
      Note: 
        Before call robust_prune, clean Nout(p) because robust_prune takes union v and Nout(p) inside.
        It may ontain deleted points.
        The original paper does not explicitly state in Algorithm 4.
      */
      self.clean_n_out_edge(p);
      self.robust_prune(p, c_with_dist);

    }

    for grave_i in self.cemetery.clone() {
      self.clean_n_out_edge(grave_i); // Backlinks are not defined in the original algorithm and should be deleted here.
    }

    // Mark node as empty
    self.empties = union_ids(&self.empties, &self.cemetery);

    self.cemetery = Vec::new();

  }


  fn pq_distance(&mut self, a: usize, b: usize) -> f32 {
    let mut dist_sum = 0.0;
    for m in 0..self.builder.pq_m {
      if let Some(dist) = self.pq_dist_map.get(&(m, a, b)) {
        dist_sum += dist;
      } else {
        let point_a = &self.codebooks[m][self.nodes[a].pq[m]];
        let point_b = &self.codebooks[m][self.nodes[b].pq[m]];
        let dist = point_a.distance(&point_b);
        self.pq_dist_map.insert((m,a,b), dist);
        dist_sum += dist;
      };

    }
    dist_sum
  }

  fn greedy_search_pq(&mut self, xq: usize, k: usize, l: usize) -> (Vec<(f32, usize)>, Vec<(f32, usize)>) { // k-anns, visited
    assert!(l >= k);


    let s = self.centroid;
    let mut visited: Vec<(f32, usize)> = Vec::new();
    // let mut list: Vec<(f32, usize)> = vec![(self.nodes[s].p.distance(xq), s)];
    let mut list: Vec<(f32, usize)> = vec![(self.pq_distance(s, xq), s)];

    // `working` is a list of unexplored candidates
    let mut working = list.clone(); // Because list\visited == list at beginning
    while working.len() > 0 {

      // let p∗ ← arg minp∈L\V ||xp − xq||
      let nearest = find_nearest(&mut working);


      // ToDo: refactoring, use union_dist insted 
      if is_contained_in(&nearest.1, &visited) {
        continue;
      } else {
        let original_dist = self.node_distance(nearest.1, xq);
        insert_dist((original_dist, nearest.1), &mut visited); // insert original dist , not quantized 
      }

      // If the node is marked as grave, remove from result list. But Its neighboring nodes are explored.
      if self.cemetery.contains(&nearest.1) {
        remove_from(&nearest, &mut list);
        // remove_from_v1(&nearest.1, &mut list)
      }


      // update L ← L ∪ Nout(p∗) andV ← V ∪ {p∗}
      for out_i in self.nodes[nearest.1].n_out.clone() {
        let out_i_point = &self.nodes[out_i].p;

        if is_contained_in(&out_i, &list) || is_contained_in(&out_i, &visited) { // Should check visited as grave point is in visited but not in list.
          continue;
        }

        let quantized_dist =  self.pq_distance(xq, out_i); // xq.distance(out_i_point);
        // list.push((dist, node_i));
        insert_dist((quantized_dist, out_i), &mut list);
      }

      if list.len() > l {
        sort_and_resize(&mut list, l)
      }

      working = set_diff(list.clone(), &visited);

    }

    sort_and_resize(&mut list, k);
    let k_anns = list;

    (k_anns, visited)

  }

  fn greedy_search(&self, xq: &P, k: usize, l: usize) -> (Vec<(f32, usize)>, Vec<(f32, usize)>) { // k-anns, visited
    assert!(l >= k);
    let s = self.centroid;
    let mut visited: Vec<(f32, usize)> = Vec::new();
    let mut list: Vec<(f32, usize)> = vec![(self.nodes[s].p.distance(xq), s)];

    // `working` is a list of unexplored candidates
    let mut working = list.clone(); // Because list\visited == list at beginning
    while working.len() > 0 {

      // let p∗ ← arg minp∈L\V ||xp − xq||
      let nearest = find_nearest(&mut working);


      // ToDo: refactoring, use union_dist insted 
      if is_contained_in(&nearest.1, &visited) {
        continue;
      } else {
        insert_dist(nearest, &mut visited);
      }

      // If the node is marked as grave, remove from result list. But Its neighboring nodes are explored.
      if self.cemetery.contains(&nearest.1) {
        remove_from(&nearest, &mut list);
        // remove_from_v1(&nearest.1, &mut list)
      }


      // update L ← L ∪ Nout(p∗) and V ← V ∪ {p∗}
      for out_i in &self.nodes[nearest.1].n_out {
        let out_i_point = &self.nodes[*out_i].p;

        if is_contained_in(out_i, &list) || is_contained_in(out_i, &visited) { // Should check visited as grave point is in visited but not in list.
          continue;
        }

        let dist = xq.distance(out_i_point);
        // list.push((dist, node_i));
        insert_dist((dist, *out_i), &mut list);
      }

      if list.len() > l {
        sort_and_resize(&mut list, l)
      }

      working = set_diff(list.clone(), &visited);

    }

    sort_and_resize(&mut list, k);
    let k_anns = list;

    (k_anns, visited)

  }

  fn robust_prune(&mut self, xp: usize, mut v: Vec<(f32, usize)>) {
    let n_out = self.nodes[xp].n_out.clone();

    // V ← (V ∪ Nout(p)) \ {p}
    for out_i in n_out {
      if !is_contained_in(&out_i, &v) {
        // let dist = node.p.distance(&self.nodes[*n_out].p);
        let dist = self.node_distance(xp, out_i);
        // v.push((dist, *n_out))
        insert_dist((dist, out_i), &mut v)
      }
    }
    remove_from(&(0.0, xp), &mut v);
    // remove_from_v1(&xp, &mut v);

    // Delete all back links of each n_out
    self.clean_n_out_edge(xp);

    // println!("v : {:?}", v);

    sort_list_by_dist(&mut v); // sort by d(p, p')


    while let Some((first, rest)) = v.split_first() {
      let (_, pa) = first.clone(); // pa is p asterisk (p*), which is nearest point to p in this loop
      // let pa_point = self.nodes[pa.1].p.clone();
      insert_id(pa, &mut self.nodes[xp].n_out);
      insert_id(xp, &mut self.nodes[pa].n_in); // back link

      if self.nodes[xp].n_out.len() == self.builder.r {
        break;
      }
      v = rest.to_vec();

      // if α · d(p*, p') <= d(p, p') then remove p' from v
      v.retain(|&(dist_xp_pd, pd)| {
        let dist_pa_pd = self.node_distance(pd, pa);
        self.builder.a * dist_pa_pd > dist_xp_pd
      })

    }

  }

  fn clean_n_out_edge(&mut self, id: usize) {
    // Delete all back links of each n_out
    let n_out = &self.nodes[id].n_out.clone();
    for out_i in n_out {
      delete_id(id, &mut self.nodes[*out_i].n_in);
    }
    self.nodes[id].n_out = vec![];
  }
}

fn set_diff(a: Vec<(f32, usize)>, b: &Vec<(f32, usize)>) -> Vec<(f32, usize)> {
  a.into_iter().filter(|(_, p)| !is_contained_in(p, b)).collect()
}

fn diff_ids(a: &Vec<usize>, b: &Vec<usize>) -> Vec<usize> {
  let mut result = Vec::new();
  let mut a_idx = 0;
  let mut b_idx = 0;

  while a_idx < a.len() && b_idx < b.len() {
    if a[a_idx] == b[b_idx] {
      a_idx += 1; // Skip common elements
      b_idx += 1;
    } else if a[a_idx] < b[b_idx] {
      // Elements present only in a
      result.push(a[a_idx]);
      a_idx += 1;
    } else {
      // Ignore elements that exist only in b
      b_idx += 1;
    }
  }

  // Add the remaining elements of a (since they do not exist in b)
  while a_idx < a.len() {
    result.push(a[a_idx]);
    a_idx += 1;
  }

  result
}

fn sort_list_by_dist(list: &mut Vec<(f32, usize)>) {
  list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less));
}

fn find_nearest(c: &mut Vec<(f32, usize)>) -> (f32, usize) {
  sort_list_by_dist(c); // ToDo: Ensure that the arugment list is already sorted.
  c[0]
}

fn sort_and_resize(list: &mut Vec<(f32, usize)>, size: usize) {
  list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less));
  list.truncate(size)
}

fn is_contained_in(i: &usize, vec: &Vec<(f32, usize)>) -> bool {
  vec.iter().filter(|(_, id)| *id == *i).collect::<Vec<&(f32, usize)>>().len() != 0
}

fn remove_from(value: &(f32, usize), vec: &mut Vec<(f32, usize)>) {
  let result = vec.binary_search_by(|probe| probe.0.partial_cmp(&value.0).unwrap());

  match result {
      Ok(index) => {
          // If an element with a matching f32 value is found, check if the usize also matches.
          // search back and forth since they may have the same f32 value.
          let mut start = index;
          while start > 0 && vec[start - 1].0 == value.0 {
              start -= 1;
          }
          let mut end = index;
          while end < vec.len() - 1 && vec[end + 1].0 == value.0 {
              end += 1;
          }

          // Find elements with matching usize values in the start to end range.
          if let Some(pos) = (start..=end).find(|&i| vec[i].1 == value.1) {
              vec.remove(pos);
          }
      },
      Err(_) => {
          // If the value of f32 is not found, nothing is done.
      },
  }
}

fn insert_id(value: usize, vec: &mut Vec<usize>) {
  match vec.binary_search(&value) {
    Ok(_index) => {
      return // If already exsits
    },
    Err(index) => {
      vec.insert(index, value);
    },
  };
}

fn delete_id(value: usize, vec: &mut Vec<usize>) {
  match vec.binary_search(&value) {
    Ok(index) => {
      vec.remove(index);
    },
    Err(_index) => {
      return
    },
  };
}

fn insert_dist(value: (f32, usize), vec: &mut Vec<(f32, usize)>) {
  match vec.binary_search_by(|probe| probe.0.partial_cmp(&value.0).unwrap_or(std::cmp::Ordering::Less)) {
    Ok(index) => {
      // identify a range of groups of elements with the same f32 value
      let mut start = index;
      while start > 0 && vec[start - 1].0 == value.0 {
          start -= 1;
      }
      let mut end = index;
      while end < vec.len() - 1 && vec[end + 1].0 == value.0 {
          end += 1;
      }

      // Check for elements with the same usize value within the specified range
      if !(start..=end).any(|i| vec[i].1 == value.1) {
          vec.insert(index, value);
      }
    },
    Err(index) => {
      vec.insert(index, value);
    },
  };
}

fn intersect_ids(a: &Vec<usize>, b: &Vec<usize>) -> Vec<usize> {
  let mut result = Vec::new();
  let mut a_idx = 0;
  let mut b_idx = 0;

  while a_idx < a.len() && b_idx < b.len() {
    if a[a_idx] == b[b_idx] {
      result.push(a[a_idx]);
      a_idx += 1;
      b_idx += 1;
    } else if a[a_idx] < b[b_idx] {
      a_idx += 1;
    } else {
      b_idx += 1;
    }
  }

  result
}

fn union_ids(a: &Vec<usize>, b: &Vec<usize>) -> Vec<usize> {
  let mut result = Vec::new();
  let mut a_idx = 0;
  let mut b_idx = 0;

  while a_idx < a.len() && b_idx < b.len() {
    if a[a_idx] == b[b_idx] {
      result.push(a[a_idx]);
      a_idx += 1;
      b_idx += 1;
    } else if a[a_idx] < b[b_idx] {
      result.push(a[a_idx]);
      a_idx += 1;
    } else {
      result.push(b[b_idx]);
      b_idx += 1;
    }
  }

  // Add the remaining elements of a or b
  while a_idx < a.len() {
    result.push(a[a_idx]);
    a_idx += 1;
  }
  while b_idx < b.len() {
    result.push(b[b_idx]);
    b_idx += 1;
  }

  result
}


pub trait Point: Clone + Sync {
    fn distance(&self, other: &Self) -> f32;
    fn dim() -> u32;
    fn to_f32_vec(&self) -> Vec<f32>;
    fn from_f32_vec(a: Vec<f32>) -> Self;
}



#[cfg(test)]
mod tests {

  use super::{Point as VPoint, *};
  use rand::rngs::SmallRng;
  use rand::SeedableRng;

  mod pq;

  use super::pq::PQ;


  #[derive(Clone, Debug)]
  struct Point(Vec<u32>);
  impl VPoint for Point {
      fn distance(&self, other: &Self) -> f32 {
          self.0.iter()
            .zip(other.0.iter())
            .map(|(a, b)| (*a as f32 - *b as f32).powi(2))
            .sum::<f32>()
            .sqrt() 
      }
      fn dim() -> u32 {
        12
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
  fn test_pq() {
    let seed: u64 = rand::random();
    // let seed: u64 = 2187599979254292977;

    println!("seed {}", seed);
    let rng = SmallRng::seed_from_u64(seed);
    let mut i = 0;
    let points: Vec<Point> = (0..100).into_iter().map(|_| {
      let a = i;
      i += 1;
      Point(vec![a; Point::dim() as usize])
    }).collect();

    let quant = PQ::new(rng, 4, 64, Point::dim() as usize, points);
    println!("{:?}", quant.quantize());
  }

  #[test]
  fn test_random_init_v2() {
    let builder = Builder::default();
    // let seed = builder.seed;
    let seed: u64 = 11923543545843533243;
    let mut rng = SmallRng::seed_from_u64(seed);
    println!("seed: {}", seed);

    let mut i = 0;

    let points: Vec<(Point, Vec<usize>)> = (0..100).into_iter().map(|_| {
      let a = i;
      i += 1;
      (Point(vec![a; Point::dim() as usize]), vec![])
    }).collect();

    let point_len = points.len();

    let ann: FreshVamana<Point> = FreshVamana::random_graph_init_v2(points, builder, &mut rng, Vec::new(), HashMap::new());
    
    for node_i in 0..point_len {
      for out_i in &ann.nodes[node_i].n_out {
        assert!(ann.nodes[*out_i].n_in.contains(&node_i))
      }
      for in_i in &ann.nodes[node_i].n_in {
        assert!(ann.nodes[*in_i].n_out.contains(&node_i))
      }
    }

  }

  #[test]
  fn fresh_disk_ann_new_empty() {
    let builder = Builder::default();
    let mut rng = SmallRng::seed_from_u64(builder.seed);

    let ann: FreshVamana<Point> = FreshVamana::random_graph_init(Vec::new(), builder, &mut rng, Vec::new(), HashMap::new());
    assert_eq!(ann.nodes.len(), 0);
  }

  #[test]
  fn fresh_disk_ann_new_centroid() {

    let builder = Builder::default();
    let mut rng = SmallRng::seed_from_u64(builder.seed);

    let mut i = 0;

    let points: Vec<Point> = (0..100).into_iter().map(|_| {
      let a = i;
      i += 1;
      Point(vec![a; Point::dim() as usize])
    }).collect();

    let (points, codebooks) = PQ::new(rng.clone(), 4, 64, Point::dim() as usize, points).quantize();

    let ann: FreshVamana<Point> = FreshVamana::random_graph_init(points, builder, &mut rng, codebooks, HashMap::new());
    assert_eq!(ann.centroid, 49);
  }

  #[test]
  fn test_vamana_build() {

    let mut builder = Builder::default();
    builder.set_l(30);
    builder.set_pq_m(4);
    // builder.set_seed(11677721592066047712);
    let l = builder.l;

    let mut i = 0;

    let points: Vec<Point> = (0..1000).into_iter().map(|_| {
      let a = i;
      i += 1;
      Point(vec![a; Point::dim() as usize])
    }).collect();

    let ann: FreshVamana<Point> = FreshVamana::new(points, builder);
    let xq = Point(vec![0; Point::dim() as usize]);
    let k = 20;
    let (k_anns, _visited) = ann.greedy_search(&xq, k, l);


    // println!("\n------- let mut ann: FreshVamana<Point> = FreshVamana::new(points, builder); --------\n");
    // for node in &ann.nodes {
    //   println!("{},  \n{:?},  \n{:?}", node.id, node.n_in, node.n_out);
    // }
    println!();

    println!("{:?}", k_anns);
    for i in 0..10 {
      assert_eq!(k_anns[i].1, i);
    }
  }


  #[test]
  fn test_greedy_search_with_cemetery() {

    let mut builder = Builder::default();
    builder.set_l(30);
    let mut rng = SmallRng::seed_from_u64(builder.seed);
    let l = builder.l;

    let mut i = 0;

    let points: Vec<Point> = (0..500).into_iter().map(|_| {
      let a = i;
      i += 1;
      Point(vec![a; Point::dim() as usize])
    }).collect();

    let (points, codebooks) = PQ::new(rng.clone(), 4, 256, Point::dim() as usize, points).quantize();

    let mut ann: FreshVamana<Point> = FreshVamana::random_graph_init(points, builder, &mut rng, codebooks, HashMap::new());

    let xq = Point(vec![0; Point::dim() as usize]);
    let k = 30;
    let (k_anns, _visited) = ann.greedy_search(&xq, k, l);

    // mark as grave
    ann.inter(k_anns[3].1);
    ann.inter(k_anns[5].1);
    ann.inter(k_anns[9].1);
    let expected = vec![k_anns[3].1, k_anns[5].1, k_anns[9].1];

    let (k_anns_intered, _visited) = ann.greedy_search(&xq, k, l);

    assert_ne!(k_anns_intered, k_anns);

    let k_anns_ids: Vec<usize>          = k_anns.into_iter().map(|(_, id)| id).collect();
    let k_anns_intered_ids: Vec<usize>  = k_anns_intered.into_iter().map(|(_, id)| id).collect();

    let diff = diff_ids(&k_anns_ids, &k_anns_intered_ids);
    assert_eq!(diff, expected);


  }

  #[test]
  fn test_greedy_search_after_removing_graves() {

    let mut builder = Builder::default();
    builder.set_l(30);
    builder.set_r(30);
    builder.set_pq_m(4);

    // builder.set_a(1.2);
    // builder.set_seed(826142338715444524);
    // let mut rng = SmallRng::seed_from_u64(builder.seed);
    let l = builder.l;

    let mut i = 0;

    let points: Vec<Point> = (0..500).into_iter().map(|_| {
      let a = i;
      i += 1;
      Point(vec![a; Point::dim() as usize])
    }).collect();

    // let mut ann: FreshVamana<Point> = FreshVamana::random_graph_init(points, builder, &mut rng);
    let mut ann: FreshVamana<Point> = FreshVamana::new(points, builder);


    println!("\n------- let mut ann: FreshVamana<Point> = FreshVamana::new(points, builder); --------\n");
    for node in &ann.nodes {
      println!("{},  \n{:?},  \n{:?}", node.id, node.n_in, node.n_out);
    }
    println!();

    let xq = Point(vec![0; Point::dim() as usize]);
    let k = 30;
    let (k_anns, _visited) = ann.greedy_search(&xq, k, l);

    // mark as grave
    ann.inter(k_anns[2].1);
    ann.inter(k_anns[5].1);
    ann.inter(k_anns[9].1);
    let expected = vec![k_anns[2].1, k_anns[5].1, k_anns[9].1];
    ann.remove_graves();

    let (k_anns_intered, _visited) = ann.greedy_search(&xq, k, l);
    // println!("{:?}\n\n{:?}", k_anns_intered, _visited);

    println!("\n------- ann.remove_graves(); --------\n");
    for node in &ann.nodes {
      println!("{},  \n{:?},  \n{:?}", node.id, node.n_in, node.n_out);
    }

    assert_ne!(k_anns_intered, k_anns);

    let k_anns_ids: Vec<usize>          = k_anns.into_iter().map(|(_, id)| id).collect();
    let k_anns_intered_ids: Vec<usize>  = k_anns_intered.into_iter().map(|(_, id)| id).collect();

    // println!("\n\n{:?}\n{:?}", k_anns_ids, k_anns_intered_ids);
    
    let diff: Vec<usize> = diff_ids(&k_anns_ids, &k_anns_intered_ids);
    assert_eq!(diff, expected);


  }

  // #[test]
  // fn test_insert_ann() {

  //   // 2520746169080459812

  //   let mut builder = Builder::default();
  //   builder.set_l(30);
  //   builder.set_r(30);
  //   builder.set_a(2.0);
  //   // builder.set_seed(14218614291317846415);
  //   // builder.set_seed(826142338715444524);
  //   // let mut rng = SmallRng::seed_from_u64(builder.seed);
  //   let l = builder.l;

  //   let mut i = 0;

  //   let points: Vec<Point> = (0..500).into_iter().map(|_| {
  //     let a = i;
  //     i += 1;
  //     Point(vec![a; Point::dim() as usize])
  //   }).collect();

  //   // let mut ann: FreshVamana<Point> = FreshVamana::random_graph_init(points, builder, &mut rng);
  //   let mut ann: FreshVamana<Point> = FreshVamana::new(points, builder);


  //   // println!("\n------- let mut ann: FreshVamana<Point> = FreshVamana::new(points, builder); --------\n");
  //   // for node in &ann.nodes {
  //   //   println!("{},  \n{:?},  \n{:?}", node.id, node.n_in, node.n_out);
  //   // }


  //   println!();

  //   let xq = Point(vec![0; Point::dim() as usize]);
  //   let k = 30;
  //   let (k_anns, _visited) = ann.greedy_search(&xq, k, l);

  //   println!("k_anns {:?}", k_anns);

  //   // mark as grave
  //   ann.inter(k_anns[2].1);
  //   ann.inter(k_anns[5].1);
  //   ann.inter(k_anns[9].1);
  //   let expected = vec![k_anns[2].1, k_anns[5].1, k_anns[9].1];
  //   let deleted = vec![ann.nodes[2].p.clone(), ann.nodes[5].p.clone(), ann.nodes[9].p.clone()];
  //   println!("expected :{:?}", expected);
  //   println!("k_anns[2].1 {}, k_anns[5].1 {}, k_anns[9].1, {}", k_anns[2].1, k_anns[5].1, k_anns[9].1);
  //   println!("deleted :{:?}", deleted);
  //   println!("cemetery :{:?}", ann.cemetery);
  //   ann.remove_graves();

  //   let (k_anns_intered, _visited) = ann.greedy_search(&xq, k, l);
  //   // println!("{:?}\n\n{:?}", k_anns_intered, _visited);

  //   // println!("\n------- ann.remove_graves(); --------\n");
  //   // for node in &ann.nodes {
  //   //   println!("{},  \n{:?},  \n{:?}", node.id, node.n_in, node.n_out);
  //   // }

  //   assert_ne!(k_anns_intered, k_anns);

  //   let k_anns_ids: Vec<usize>          = k_anns.clone().into_iter().map(|(_, id)| id).collect();
  //   let k_anns_intered_ids: Vec<usize>  = k_anns_intered.into_iter().map(|(_, id)| id).collect();

  //   // println!("\n\n{:?}\n{:?}", k_anns_ids, k_anns_intered_ids);
    
  //   let diff: Vec<usize> = diff_ids(&k_anns_ids, &k_anns_intered_ids);
  //   assert_eq!(diff, expected);

  //   for d in deleted {
  //     ann.insert(d)
  //   }

  //   let (k_anns_inserted, _) = ann.greedy_search(&xq, k, l);
  //   assert_eq!(k_anns_inserted, k_anns);
  //   println!("{:?}", k_anns_inserted);


  // }


  #[test]
  fn greedy_search() {

    let mut builder = Builder::default();
    builder.set_l(30);
    println!("seed: {}", builder.seed);
    let seed = builder.seed;
    // let seed: u64 = 6752150918298254033;
    let mut rng = SmallRng::seed_from_u64(seed);
    let l = builder.l;

    let mut i = 0;

    let points: Vec<Point> = (0..500).into_iter().map(|_| {
      let a = i;
      i += 1;
      Point(vec![a; Point::dim() as usize])
    }).collect();

    let (points, codebooks) = PQ::new(rng.clone(), 4, 256, Point::dim() as usize, points).quantize();

    let ann: FreshVamana<Point> = FreshVamana::random_graph_init(points, builder, &mut rng, codebooks, HashMap::new());
    let xq = Point(vec![0; Point::dim() as usize]);
    let k = 10;
    let (k_anns, _visited) = ann.greedy_search(&xq, k, l);

    println!("k_anns: {:?}", k_anns);

    for i in 0..10 {
      assert_eq!(k_anns[i].1, i);
    }
  }

  #[test]
  fn test_robust_prune() {

    let mut builder = Builder::default();
    builder.set_l(30);
    let l = builder.l;
    let mut rng = SmallRng::seed_from_u64(builder.seed);

    let mut i = 0;

    let points: Vec<Point> = (0..500).into_iter().map(|_| {
      let a = i;
      i += 1;
      Point(vec![a; Point::dim() as usize])
    }).collect();

    let i = 11;
    let xq = &points[i].clone();

    let (points, codebooks) = PQ::new(rng.clone(), 4, 256, Point::dim() as usize, points).quantize();

    let mut ann: FreshVamana<Point> = FreshVamana::random_graph_init(points.clone(), builder, &mut rng, codebooks, HashMap::new());
    let prev_n_out = ann.nodes[i].n_out.clone();
    let k = 1;
    let (_k_anns, visited) = ann.greedy_search(xq, k, l);

    ann.robust_prune(i, visited);
    let pruned_n_out = &ann.nodes[i].n_out;
    assert_ne!(prev_n_out, *pruned_n_out);

    // let dist_pa_pd =  self.nodes[pd].p.distance(&pa_point);
    // self.builder.a * dist_pa_pd > dist_xp_pd

    let mut v = pruned_n_out.clone();
    while let Some((pa, rest)) = v.split_first() {
      if rest.len() == 0 {break}

      let pd = rest[0];

      // let dist_xp_pd = ann.nodes[pd].p.distance(&xq); (ann.node_distance(out_i, j), out_i)
      // let dist_pa_pd = ann.nodes[pd].p.distance(&ann.nodes[*pa].p);

      let dist_xp_pd = ann.node_distance(pd, i);
      let dist_pa_pd = ann.node_distance(pd, *pa);


      assert!(ann.builder.a * dist_pa_pd > dist_xp_pd);

      v = rest.to_vec();
    }
  }

  #[test]
  fn test_set_diff() {
    let a = vec![(0.0, 0), (0.1, 1), (0.2, 2), (0.3, 3)];
    let b = vec![(0.0, 0), (0.1, 1)];

    let c = set_diff(a, &b);
    assert_eq!(c, vec![(0.2, 2), (0.3, 3)])
  }

  #[test]
  fn test_sort_list_by_dist() {
    let mut a = vec![(0.2, 2), (0.1, 1), (0.3, 3), (0.0, 0)];
    sort_list_by_dist(&mut a);
    assert_eq!(a, vec![(0.0, 0), (0.1, 1), (0.2, 2), (0.3, 3)])
  }

  #[test]
  fn test_find_nearest() {
    let mut a = vec![(0.2, 2), (0.1, 1), (0.3, 3), (0.0, 0)];
    assert_eq!(find_nearest(&mut a), (0.0, 0));
  }

  #[test]
  fn test_sort_and_resize() {
    let mut a = vec![(0.2, 2), (0.1, 1), (0.3, 3), (0.0, 0)];
    sort_and_resize(&mut a, 2);
    assert_eq!(a, vec![(0.0, 0), (0.1, 1)])
  }

  #[test]
  fn test_is_contained_in() {
    let a = vec![(0.2, 2), (0.1, 1), (0.3, 3), (0.0, 0)];
    assert!(is_contained_in(&0, &a));
    assert!(!is_contained_in(&10, &a));
  }

  #[test]
  fn test_remove_from() {
    let mut a = vec![(0.2, 2), (0.1, 1), (0.3, 3), (0.0, 0)];
    remove_from(&(0.3, 3), &mut a);
    assert_eq!(a, vec![(0.2, 2), (0.1, 1), (0.0, 0)])

  }

  #[test]
  fn test_insert_id() {
    let mut a = vec![0, 1 , 3 , 4];
    insert_id(2, &mut a);
    insert_id(2, &mut a);
    assert_eq!(a, vec![0, 1 , 2 , 3, 4]);

    let mut a = vec![1 , 3 , 4];
    insert_id(0, &mut a);
    assert_eq!(a, vec![0, 1 , 3, 4])

  }

  #[test]
  fn test_insert_dist() {
    let mut a = vec![(0.0, 0), (0.1, 1), (0.3, 3)];
    insert_dist((0.2, 2), &mut a);
    insert_dist((0.2, 2), &mut a);
    assert_eq!(a, vec![(0.0, 0), (0.1, 1), (0.2, 2), (0.3, 3)]);

    let mut a = vec![(0.0, 1), (1.7320508, 2), (3.4641016, 3), (5.196152, 4), (6.928203, 5), (8.6602545, 6), (12.124355, 8), (13.856406, 9), (15.588457, 10), (17.320509, 11), (19.052559, 12), (20.784609, 13), (22.51666, 14), (24.24871, 15), (27.712812, 17), (862.5613, 499)];
    insert_dist((1.7320508, 0), &mut a);
    assert_eq!(a, vec![(0.0, 1), (1.7320508, 0), (1.7320508, 2), (3.4641016, 3), (5.196152, 4), (6.928203, 5), (8.6602545, 6), (12.124355, 8), (13.856406, 9), (15.588457, 10), (17.320509, 11), (19.052559, 12), (20.784609, 13), (22.51666, 14), (24.24871, 15), (27.712812, 17), (862.5613, 499)])

  }

  #[test]
  fn test_intersect_ids() {
    let a = vec![0, 1 , 3 , 4];
    let c = intersect_ids(&a,&a);
    assert_eq!(c, a);

    let b = vec![0, 4];
    let c = intersect_ids(&a, &b);
    assert_eq!(c, b);

    let b = vec![0, 2, 4];
    let c = intersect_ids(&a, &b);
    assert_eq!(c, vec![0, 4]);
  }

  #[test]
  fn test_diff_ids() {
    let a = vec![0, 1 , 3 , 4];
    let b = vec![0, 4];
    let c = diff_ids(&a, &b);
    assert_eq!(c, vec![1, 3]);

    let b = vec![0, 4, 5];
    let c = diff_ids(&a, &b);
    assert_eq!(c, vec![1, 3]);

    let b = vec![0, 1 , 3 , 4];
    let c = diff_ids(&a, &b);
    // assert_eq!(c, vec![]);

    let b = vec![];
    let c = diff_ids(&a, &b);
    assert_eq!(c, a);
  }

  #[test]
  fn test_union_ids() {
    let a = vec![0, 1 , 3 , 4];
    let b = vec![0, 4];
    let c = union_ids(&a, &b);
    assert_eq!(c, a);

    let b = vec![0, 4, 5];
    let c = union_ids(&a, &b);
    assert_eq!(c, vec![0, 1 , 3 , 4, 5]);

    // let b = vec![0, 1 , 3 , 4];
    // let c = diff_ids(&a,&b);
    // assert_eq!(c, vec![]);

    // let b = vec![];
    // let c = diff_ids(&a,&b);
    // assert_eq!(c, a);
  }
}
