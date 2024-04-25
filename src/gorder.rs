use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, Ordering};

fn pack_node(
  original_index: &u32,
  shuffled_nodes: &Vec<(u32, AtomicBool, &Vec<u32>)>,
  shuffle_table: &Vec<usize>,
) -> bool {
  let shuffled_index = shuffle_table[*original_index as usize];
  let packed_flag = &shuffled_nodes[shuffled_index].1;
  
  packed_flag.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok()
}

fn select_random_s(shuffled_nodes: &Vec<(u32, AtomicBool, &Vec<u32>)>) -> Result<u32, ()> {
  let mut scan_index = 0;
  loop {
      let packed_flag = &shuffled_nodes[scan_index].1;
      match packed_flag.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst) {
          Ok(_) => {
              let original_index = &shuffled_nodes[scan_index].0;
              return Ok(*original_index);
          }
          Err(_) => {
              scan_index += 1;
          }
      }
      if scan_index == shuffled_nodes.len() {
          return Err(());
      }
  }
}

fn sector_packing(
  window_size: usize,
  nodes: &Vec<Vec<u32>>,
  backlinks: &Vec<Vec<u32>>,
  shuffled_nodes: &Vec<(u32, AtomicBool, &Vec<u32>)>,
  shuffle_table: &Vec<usize>,
) -> Vec<u32> {
  let mut sub_array = Vec::with_capacity(window_size);
  let mut sub_array_index = 0;
  let mut heap = KeyMaxHeap::new();

  // Pick a random, unpacked seed node s.
  sub_array.push(select_random_s(shuffled_nodes).unwrap());

  while sub_array_index < window_size {
      // 𝑣𝑒 ← 𝑃 [𝑖];𝑖 ← 𝑖 + 1
      let ve = &sub_array[sub_array_index];
      sub_array_index += 1;

      // for 𝑢 ∈ 𝑁out(𝑣𝑒 ) do
      //   H.IncrementKey(𝑢)
      for u in &nodes[*ve as usize] {
          heap.increment_key(*u);
      }

      //  for 𝑢 ∈ 𝑁in (𝑣𝑒 ) do
      //    H.IncrementKey(𝑢)
      //      for 𝑡 ∈ 𝑁out(𝑢) do
      //        H.IncrementKey(𝑡)
      for u in &backlinks[*ve as usize] {
          heap.increment_key(*u);
          for t in &nodes[*u as usize] {
              heap.increment_key(*t);
          }
      }

      let v_max = loop {
          match heap.get_max() {
              None => {
                  // if H.empty() then Pick a random unpacked seed node 𝑣max and break.
                  match select_random_s(shuffled_nodes) {
                      Ok(s) => {
                          break s;
                      }
                      Err(_) => {
                          // If no unpacked nodes are found, Sector Packing is returned.
                          return sub_array;
                      }
                  }
              }
              Some((v_max_candidate_index, _)) => {
                  // if not D [𝑣max ] then break
                  if pack_node(&v_max_candidate_index, shuffled_nodes, shuffle_table) {
                      break v_max_candidate_index;
                  }
              }
          }
      };
      sub_array.push(v_max);
  }

  sub_array
}

///
/// Reordering the arrangement to efficiently reference nodes from storage such as SSDs.
/// This algorithm is proposed in Section 4 of this [paper](https://arxiv.org/pdf/2211.12850v2.pdf). 
///
pub fn gorder(
  nodes: Vec<Vec<u32>>,
  backlinks: Vec<Vec<u32>>,
  window_size: usize,
  rng: &mut SmallRng,
) -> Vec<u32> {
  /* Parallel Gordering */
  // Select unpacked node randomly.
  // Scan from end to end to find nodes with the packed flag false and pick the first unpacked node found.
  // The nodes are shuffled to ensure that start nodes are randomly selected.
  let mut shuffled_nodes: Vec<(u32, AtomicBool, &Vec<u32>)> = nodes
      .iter()
      .enumerate()
      .map(|(original_idx, n_out)| (original_idx as u32, AtomicBool::new(false), n_out))
      .collect();
  shuffled_nodes.shuffle(rng);
  let shuffle_table: Vec<usize> = shuffled_nodes
      .iter()
      .enumerate()
      .map(|(shuffled_index, (original_idx, _, _))| (*original_idx, shuffled_index))
      .sorted()
      .map(|(_, shuffled_index)| shuffled_index)
      .collect();

  // parallel for 𝑖 ∈ [0, 1, . . . , ⌊|X|/𝑤⌋ − 1] do
  //   Pick a random, unpacked seed node 𝑠.
  //   SectorPack(𝑃 [𝑖 ∗ 𝑤], D, 𝑠, 𝑤,)
  let mut reordered: Vec<u32> = (0..(nodes.len() / window_size) - 1)
      .into_par_iter()
      // .into_iter()
      .map(|_start_array_position: usize| {
          sector_packing(
              window_size,
              &nodes,
              &backlinks,
              &shuffled_nodes,
              &shuffle_table,
          )
      })
      .flatten()
      .collect();

  // Pick a random, unpacked seed node 𝑠.
  // SectorPack(𝑃 [ ⌊ |X|/𝑤⌋ ∗ 𝑤], D, 𝑠, 𝑤,)
  reordered.extend(sector_packing(
      window_size,
      &nodes,
      &backlinks,
      &shuffled_nodes,
      &shuffle_table,
  ));

  reordered
}

struct KeyMaxHeap {
  heap: BinaryHeap<(u32, u32)>,
  counts: FxHashMap<u32, u32>,
}

impl KeyMaxHeap {
  fn new() -> Self {
      let mut counts = FxHashMap::default();
      counts.reserve(100);
      Self {
          heap: BinaryHeap::new(),
          counts,
      }
  }

  fn increment_key(&mut self, v: u32) {
      let count = *self.counts.get(&v).unwrap_or(&0) + 1;
      self.heap.push((count, v));
      self.counts.insert(v, count);
  }

  fn get_max(&mut self) -> Option<(u32, u32)> {
      while let Some((count, v)) = self.heap.pop() {
          if let Some(&stored_count) = self.counts.get(&v) {
              if count == stored_count {
                  return Some((v, count));
              }
          }
      }
      None
  }
}