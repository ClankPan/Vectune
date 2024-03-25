use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/*

cache戦略のアイデア
  ICはSSDのような並列に読み出しできない環境では、空間的に近いノードを近くに配置しておくことが望ましい。
  insertで、最近傍を探して、最も近い空いてる場所に入れるような方式をとるのはどうか。
  10%多く見積もって、あらかじめ隙間を開けておく。適度に圧縮・拡張をする。

*/

/*
Debug Note:
  - After indexing three times, the graph is quite healthy. Figure out why it is.
*/

/*
Note API
  - Prepare insert and union separately as api.　
    The insert allows duplicates, while the union does not insert if the file already exists.
*/


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
      a: 2.0,
      r: 70,
      l: 125,
      seed: rand::random(),
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

  pub fn build<P: Point, V: Clone>(self, points: Vec<P>, values: Vec<V>) -> FreshVamanaMap<P, V>{
    FreshVamanaMap::new(points, values, self)
  }
}



pub struct FreshVamanaMap<P, V> {
  ann: FreshVamana<P>,
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

struct Node<P> {
  n_out: Vec<usize>, // has pointer. ToDo: should use PQ to resuce memory accesses.
  n_in: Vec<usize>,
  p: P,
  id: usize,
}

impl<P> Node<P> {
  fn new(p: P, id: usize) -> Self {
    Self {
      n_out: Vec::new(),
      n_in: Vec::new(),
      p,
      id,
    }
  }
}

pub struct FreshVamana<P>
{
  nodes: Vec<Node<P>>,
  centroid: usize,
  builder: Builder,
  cemetery: Vec<usize>,
  empties: Vec<usize>,
}

impl<P> FreshVamana<P>
where
    P: Point,
{
  pub fn new(points: Vec<P>, builder: Builder) -> Self {
    let mut rng = SmallRng::seed_from_u64(builder.seed);
    println!("seed: {}", builder.seed);

    // Initialize Random Graph
    let mut ann = FreshVamana::<P>::random_graph_init(points, builder, &mut rng);


    // Prune Edges

    // let σ denote a random permutation of 1..n
    let node_len = ann.nodes.len();
    let mut shuffled: Vec<(usize, usize)> = (0..node_len).into_iter().map(|node_i| (rng.gen_range(0..node_len as usize), node_i)).collect();
    shuffled.sort_by(|a, b| a.0.cmp(&b.0));

    FreshVamana::<P>::indexing(&mut ann, shuffled);


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



    ann
  }

  fn indexing( ann: &mut FreshVamana<P>, shuffled: Vec<(usize, usize)>) {

    // for 1 ≤ i ≤ n do
    for (count, (_, i)) in shuffled.into_iter().enumerate() {

      println!("id : {}\t/{}", count, ann.nodes.len());

      // let [L; V] ← GreedySearch(s, xσ(i), 1, L)
      let (_, visited) = ann.greedy_search(&ann.nodes[i].p, 1, ann.builder.l);
      // println!("visited: {:?}", visited.clone().iter().map(|(_, i)| *i).collect::<Vec<usize>>());
      // run RobustPrune(σ(i), V, α, R) to update out-neighbors of σ(i)
      ann.robust_prune(i, visited);
      // println!("n_out: {:?}", &ann.nodes[i].n_out);

      // for all points j in Nout(σ(i)) do
      for j in ann.nodes[i].n_out.clone() {
        if ann.nodes[j].n_out.contains(&i) {
          continue;
        } else {
          // Todo : refactor, self.make_edge　or union. above ann.nodes[j].n_out.contains(&i) not necessary if use union
          insert_id(i, &mut ann.nodes[j].n_out);
          insert_id(j, &mut ann.nodes[i].n_in);
        }

        let j_point = &ann.nodes[j].p;

        // if |Nout(j) ∪ {σ(i)}| > R then run RobustPrune(j, Nout(j) ∪ {σ(i)}, α, R) to update out-neighbors of j
        if ann.nodes[j].n_out.len() > ann.builder.r {
          // robust_prune requires (dist(xp, p'), index)
          let v: Vec<(f32, usize)> = ann.nodes[j].n_out.clone().into_iter()
            .map(|out_i: usize| 
              (ann.nodes[out_i].p.distance(j_point), out_i)
            ).collect();

          ann.robust_prune(j, v);
        }
      }
    }
  }

  pub fn insert(&mut self, p: P) {
    // Add node
    let pid =  if self.empties.len() == 0 { // ToDo: cache
      let id = self.nodes.len();
      self.nodes.push(Node::new(p.clone(), id));
      id
    } else {
      let id = self.empties[0];
      self.empties.remove(0);
      self.nodes[id] = Node::new(p.clone(), id);
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

  fn make_edge(out_i: usize, in_i: usize) { // out_i -> in_i
    // todo make sure adding backlinks
    todo!();
  }

  pub fn remove_graves(&mut self) {
    // for node in &self.nodes {
    //   // println!("id {},\n out: {:?},\n in: {:?}", node.id, node.n_out, node.n_in);
    // }
    // println!("");


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

      // println!("id: {}, D {:?} , C: {:?}", p, d, c);

      // C ← C \ D
      /*
       Note:
        Since D's Nout may contain LD, Why pull the D instead of the LD?
        I implemented it as shown and it hit data that should have been erased, so I'll fix it to pull LD.
        `c = diff_ids(&c, &d); // <- ???`
      */
      c = diff_ids(&c, &self.cemetery);

      // println!("id: {}, c {:?}, u: {:?}", p, c, d.iter().map(|u| self.nodes[*u].n_out.clone()).collect::<Vec<Vec<usize>>>());

      // 𝑁out(𝑝) ← RobustPrune(𝑝, C, 𝛼, 𝑅)
      let p_point = self.nodes[p].p.clone();
      let mut c_with_dist: Vec<(f32, usize)> = c.into_iter().map(|id| (p_point.distance(&self.nodes[id].p), id)).collect();

      // println!("id: {}, c {:?}", p, c_with_dist);

      sort_list_by_dist(&mut c_with_dist);
      /* 
      Note: 
        Before call robust_prune, clean Nout(p) because robust_prune takes union v and Nout(p) inside.
        It may ontain deleted points.
        The original paper does not explicitly state in Algorithm 4.
      */
      self.clean_n_out_edge(p);
      self.robust_prune(p, c_with_dist);

      // println!("n_out {:?}", self.nodes[p].n_out);
    }

    for grave_i in self.cemetery.clone() {
      self.clean_n_out_edge(grave_i); // Backlinks are not defined in the original algorithm and should be deleted here.
    }

    // Mark node as empty
    self.empties = union_ids(&self.empties, &self.cemetery);

    self.cemetery = Vec::new();

  }

  fn random_graph_init(points: Vec<P>, builder: Builder, rng: &mut SmallRng) -> Self {

    if points.is_empty() {
      return Self {
          nodes: Vec::new(),
          centroid: usize::MAX,
          builder,
          cemetery: Vec::new(),
          empties: Vec::new(),
        }
    }

    assert!(points.len() < u32::MAX as usize);
    let points_len = points.len();

    /* Find Centroid */
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


    /* Get random connected graph */
    let mut nodes: Vec<Node<P>> = points.into_iter().enumerate().map(|(id, p)| Node {
      n_out: Vec::new(),
      n_in: Vec::new(),
      p,
      id
    }).collect();
    
    let mut working: Vec<usize> = (0..nodes.len()).collect();
    let node_len = nodes.len();

    for node_i in 0..node_len {
      // let mut outs: Vec<usize> = Vec::new();
      // while outs.len() < builder.r {
      //   let rand_node_i = rng.gen_range(0..working.len() as usize);
      //   if rand_node_i == node_i {
      //     continue;
      //   }
      //   outs = union_ids(&outs, &vec![rand_node_i]);
      // }

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


    // println!("----------- Init Rand Graph -----------");
    // for node in &nodes {
    //   println!("{},  \n{:?},  \n{:?}", node.id, node.n_in, node.n_out);
    // }

    Self {
      nodes,
      centroid,
      builder,
      cemetery: Vec::new(),
      empties: Vec::new(),
    }

  }

  fn greedy_search(&self, xq: &P, k: usize, l: usize) -> (Vec<(f32, usize)>, Vec<(f32, usize)>) { // k-anns, visited
    assert!(l >= k);
    let s = self.centroid;
    let mut visited: Vec<(f32, usize)> = Vec::new();
    let mut list: Vec<(f32, usize)> = vec![(self.nodes[s].p.distance(xq), s)];

    // `working` is a list of unexplored candidates
    let mut working = list.clone(); // Because list\visited == list at beginning
    while working.len() > 0 {

      // println!("list: {:?}, visited: {:?} \n\n\n", list, visited);

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
      // println!("\nvisited: {:?}\nn_out of {}: {:?}", visited, nearest.1, self.nodes[nearest.1].n_out);


      // update L ← L ∪ Nout(p∗) andV ← V ∪ {p∗}
      for out_i in &self.nodes[nearest.1].n_out {
        let out_i_point = &self.nodes[*out_i].p;

        if is_contained_in(out_i, &list) || is_contained_in(out_i, &visited) { // Should check visited as grave point is in visited but not in list.
          continue;
        }

        let dist = xq.distance(out_i_point);
        // list.push((dist, node_i));
        insert_dist((dist, *out_i), &mut list);
      }

      // println!("visited: {:?}\nout_i of nearest: {:?}", visited, self.nodes[nearest.1].n_out);

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
    let node = &self.nodes[xp];

    // V ← (V ∪ Nout(p)) \ {p}
    for n_out in &node.n_out {
      if !is_contained_in(n_out, &v) {
        let dist = node.p.distance(&self.nodes[*n_out].p);
        // v.push((dist, *n_out))
        insert_dist((dist, *n_out), &mut v)
      }
    }
    remove_from(&(0.0, xp), &mut v);
    // remove_from_v1(&xp, &mut v);

    // Delete all back links of each n_out
    // let n_out = &self.nodes[xp].n_out.clone();
    // for out_i in n_out {
    //   self.nodes[*out_i].n_in.retain(|&x| x!=xp);
    // }
    // self.nodes[xp].n_out = vec![];
    self.clean_n_out_edge(xp);

    // println!("v : {:?}", v);

    sort_list_by_dist(&mut v); // sort by d(p, p')


    while let Some((first, rest)) = v.split_first() {
      let pa = first; // pa is p asterisk (p*), which is nearest point to p in this loop
      let pa_point = self.nodes[pa.1].p.clone();
      insert_id(pa.1, &mut self.nodes[xp].n_out);
      insert_id(xp, &mut self.nodes[pa.1].n_in); // back link

      if self.nodes[xp].n_out.len() == self.builder.r {
        break;
      }
      v = rest.to_vec();

      // if α · d(p*, p') <= d(p, p') then remove p' from v
      // v.retain(|&(dist_xp_pd, pd)| { // pd is p-dash (p')
      //   let dist_pa_pd =  self.nodes[pd].p.distance(&pa_point);
      //   self.builder.a * dist_pa_pd > dist_xp_pd
      // });
      let mut v_new = vec![(0.0, 0); v.len()];
      let mut v_new_count = 0;
      for (dist_xp_pd, pd) in v {
          let dist_pa_pd =  self.nodes[pd].p.distance(&pa_point);
          if self.builder.a * dist_pa_pd > dist_xp_pd {
            v_new.push((dist_xp_pd, pd));
            v_new_count += 1;
          }
      }
      v_new.truncate(v_new_count);
      v = v_new;
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

// fn remove_from_v1(i: &usize, vec: &mut Vec<(f32, usize)>) {
//   vec.retain(|&(_, x)| x!=*i);
// }

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
  use rand::{Rng, SeedableRng};


  #[derive(Clone, Debug)]
  struct Point(Vec<u32>);
  impl VPoint for Point {
      fn distance(&self, other: &Self) -> f32 {
          self.0.iter()
            .zip(other.0.iter())
            .map(|(a, b)| (*a as f32 - *b as f32).powi(2))
            .sum::<f32>()
            // .sqrt()
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
    let mut rng = SmallRng::seed_from_u64(builder.seed);

    let ann: FreshVamana<Point> = FreshVamana::random_graph_init(Vec::new(), builder, &mut rng);
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
      Point(vec![a;3])
    }).collect();

    let ann: FreshVamana<Point> = FreshVamana::random_graph_init(points, builder, &mut rng);
    assert_eq!(ann.centroid, 49);
  }

  #[test]
  fn test_vamana_build() {

    let mut builder = Builder::default();
    builder.set_l(30);
    // builder.set_seed(11677721592066047712);
    let l = builder.l;

    let mut i = 0;

    let points: Vec<Point> = (0..1000).into_iter().map(|_| {
      let a = i;
      i += 1;
      Point(vec![a;3])
    }).collect();

    let ann: FreshVamana<Point> = FreshVamana::new(points, builder);
    let xq = Point(vec![0;3]);
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

    let points: Vec<Point> = (0..100).into_iter().map(|_| {
      let a = i;
      i += 1;
      Point(vec![a;3])
    }).collect();

    let mut ann: FreshVamana<Point> = FreshVamana::random_graph_init(points, builder, &mut rng);

    let xq = Point(vec![0;3]);
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
    builder.set_a(1.2);
    // builder.set_seed(826142338715444524);
    // let mut rng = SmallRng::seed_from_u64(builder.seed);
    let l = builder.l;

    let mut i = 0;

    let points: Vec<Point> = (0..500).into_iter().map(|_| {
      let a = i;
      i += 1;
      Point(vec![a;3])
    }).collect();

    // let mut ann: FreshVamana<Point> = FreshVamana::random_graph_init(points, builder, &mut rng);
    let mut ann: FreshVamana<Point> = FreshVamana::new(points, builder);


    println!("\n------- let mut ann: FreshVamana<Point> = FreshVamana::new(points, builder); --------\n");
    for node in &ann.nodes {
      println!("{},  \n{:?},  \n{:?}", node.id, node.n_in, node.n_out);
    }
    println!();

    let xq = Point(vec![0;3]);
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

  #[test]
  fn test_insert_ann() {

    let mut builder = Builder::default();
    builder.set_l(30);
    builder.set_r(30);
    builder.set_a(1.2);
    // builder.set_seed(826142338715444524);
    // let mut rng = SmallRng::seed_from_u64(builder.seed);
    let l = builder.l;

    let mut i = 0;

    let points: Vec<Point> = (0..500).into_iter().map(|_| {
      let a = i;
      i += 1;
      Point(vec![a;3])
    }).collect();

    // let mut ann: FreshVamana<Point> = FreshVamana::random_graph_init(points, builder, &mut rng);
    let mut ann: FreshVamana<Point> = FreshVamana::new(points, builder);


    println!("\n------- let mut ann: FreshVamana<Point> = FreshVamana::new(points, builder); --------\n");
    for node in &ann.nodes {
      println!("{},  \n{:?},  \n{:?}", node.id, node.n_in, node.n_out);
    }
    println!();

    let xq = Point(vec![0;3]);
    let k = 30;
    let (k_anns, _visited) = ann.greedy_search(&xq, k, l);

    // mark as grave
    ann.inter(k_anns[2].1);
    ann.inter(k_anns[5].1);
    ann.inter(k_anns[9].1);
    let expected = vec![k_anns[2].1, k_anns[5].1, k_anns[9].1];
    let deleted = vec![ann.nodes[2].p.clone(), ann.nodes[5].p.clone(), ann.nodes[9].p.clone()];
    ann.remove_graves();

    let (k_anns_intered, _visited) = ann.greedy_search(&xq, k, l);
    // println!("{:?}\n\n{:?}", k_anns_intered, _visited);

    println!("\n------- ann.remove_graves(); --------\n");
    for node in &ann.nodes {
      println!("{},  \n{:?},  \n{:?}", node.id, node.n_in, node.n_out);
    }

    assert_ne!(k_anns_intered, k_anns);

    let k_anns_ids: Vec<usize>          = k_anns.clone().into_iter().map(|(_, id)| id).collect();
    let k_anns_intered_ids: Vec<usize>  = k_anns_intered.into_iter().map(|(_, id)| id).collect();

    // println!("\n\n{:?}\n{:?}", k_anns_ids, k_anns_intered_ids);
    
    let diff: Vec<usize> = diff_ids(&k_anns_ids, &k_anns_intered_ids);
    assert_eq!(diff, expected);

    for d in deleted {
      ann.insert(d)
    }

    let (k_anns_inserted, _) = ann.greedy_search(&xq, k, l);
    assert_eq!(k_anns_inserted, k_anns);
    println!("{:?}", k_anns_inserted);


  }


  #[test]
  fn greedy_search() {

    let mut builder = Builder::default();
    builder.set_l(30);
    let mut rng = SmallRng::seed_from_u64(builder.seed);
    let l = builder.l;

    let mut i = 0;

    let points: Vec<Point> = (0..100).into_iter().map(|_| {
      let a = i;
      i += 1;
      Point(vec![a;3])
    }).collect();

    let ann: FreshVamana<Point> = FreshVamana::random_graph_init(points, builder, &mut rng);
    let xq = Point(vec![0;3]);
    let k = 10;
    let (k_anns, _visited) = ann.greedy_search(&xq, k, l);

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

    let points: Vec<Point> = (0..100).into_iter().map(|_| {
      let a = i;
      i += 1;
      Point(vec![a;3])
    }).collect();

    let i = 11;
    let xq = &points[i];

    let mut ann: FreshVamana<Point> = FreshVamana::random_graph_init(points.clone(), builder, &mut rng);
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

      let dist_xp_pd = ann.nodes[pd].p.distance(&xq);
      let dist_pa_pd = ann.nodes[pd].p.distance(&ann.nodes[*pa].p);

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
    assert_eq!(c, vec![]);

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
