use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::future::Future;

use itertools::Itertools;
use log::debug;
use rand::Rng;
use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;

pub use crate::traits::GraphInterface;
pub use crate::traits::PointInterface;

// Partial orderなものをTotal orderにする
#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct Dist<T>(pub T);

impl<T: PartialEq> Eq for Dist<T> {}

impl<T: PartialOrd> Ord for Dist<T> {
    fn cmp(&self, other: &Dist<T>) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).expect("NaN was compared.")
    }
}

#[derive(Clone)]
pub struct Node<P: PointInterface, const R: usize> {
    pub point: P,
    pub edges: [u32; R],
}

impl<P, const R: usize> Node<P, R>
where
    P: PointInterface,
{
    pub fn new(point: P, edges: [u32; R]) -> Self {
        Self { point, edges }
    }

    pub fn point(&self) -> P {
        self.point.clone()
    }

    pub fn edges(&self) -> Vec<u32> {
        let raw_edges = self.raw_edges();
        raw_edges.into_iter().sorted().dedup().collect()
    }

    pub fn raw_edges(&self) -> [u32; R] {
        self.edges
    }
}

pub trait StorageTrait<P: PointInterface, const R: usize, T: Rng> {
    fn init() -> Self;

    fn get(&self, index: &u32) -> Option<Node<P, R>>;

    fn set(&mut self, index: u32, node: Node<P, R>) -> Option<Node<P, R>>;

    fn delete(&mut self, index: u32);

    fn remove(&mut self, index: u32);

    fn removed_indices(&self) -> Vec<u32>;

    fn clear_removed_indices(&mut self) -> Vec<u32>;

    fn alloc(&mut self) -> u32;

    fn pick_random_active_node(&self, rng: &mut T) -> (u32, Node<P, R>);

    fn backlink(&self, index: u32) -> Vec<u32>;
}

pub struct Graph<P: PointInterface, S: StorageTrait<P, R, T>, const R: usize, T>
where
    T: Rng,
{
    storage: S,
    phantom1: std::marker::PhantomData<P>,
    phantom2: std::marker::PhantomData<T>,
}

impl<P, S, const R: usize, T> Graph<P, S, R, T>
where
    P: PointInterface,
    S: StorageTrait<P, R, T>,
    T: Rng,
{
    // If this already contains a Node 0, then that
    // map is loaded. Otherwise, a new storage is created.
    pub fn init(dim: usize) -> Self {
        let mut storage = S::init();
        match storage.get(&0) {
            Some(_) => {}
            None => {
                let init_node = Node {
                    point: P::from_f32_vec(vec![0.0; dim]),
                    edges: [0; R],
                };
                storage.set(0, init_node);
            }
        }
        Self {
            storage,
            phantom1: std::marker::PhantomData,
            phantom2: std::marker::PhantomData,
        }
    }

    pub fn search(&self, query: &P, l: usize, rng: &mut T) -> Vec<(Dist<f32>, u32)> {
        let mut result = Vec::new();
        // “visited” tracks which nodes we have *finished* processing
        let mut visited_nodes = FxHashSet::default();
        // “touched” tracks which nodes we have *pushed* into the heap at least once
        let mut touched_nodes = FxHashSet::default();

        // A min-heap is emulated by using Reverse(...) in Rust's max-heap
        let mut heap = BinaryHeap::new();

        // removed removed_indices
        let removed_indices = self.storage.removed_indices();

        // Pick a random active node
        let (entry_node_index, entry_node) = self.storage.pick_random_active_node(rng);
        let entry_dist = Dist(entry_node.point.distance(query));

        let mut cache = FxHashMap::default();
        cache.insert(entry_node_index, entry_node);

        // Push the starting node onto the heap
        heap.push(Reverse((entry_dist, entry_node_index)));
        touched_nodes.insert(entry_node_index);

        // Repeatedly pop the closest unvisited node
        while let Some(Reverse((dist, node_index))) = heap.pop() {
            // If we’ve never visited it before, process it
            if visited_nodes.insert(node_index) {
                let Some(working_node): Option<&Node<P, R>> = cache.get(&node_index) else {
                    panic!("working node should be active");
                };

                // If the node is alive, add it to our result set
                if !removed_indices.contains(&node_index) {
                    result.push((dist, node_index));
                }

                // Whether alive or not, we still explore its edges
                for edge_node_index in working_node.edges() {
                    // Only push if we haven't “touched” it before
                    if touched_nodes.insert(edge_node_index) {
                        let Some(edge_node) = self.storage.get(&edge_node_index) else {
                            panic!("edge node should be active");
                        };
                        let edge_dist = Dist(edge_node.point().distance(query));
                        cache.insert(edge_node_index, edge_node);
                        // Push neighbor into the heap
                        heap.push(Reverse((edge_dist, edge_node_index)));
                    }
                }

                // If we only care about the first `l` alive nodes, we can stop early
                if result.len() >= l {
                    break;
                }
            }
        }

        // You may want them strictly sorted by distance in the final output
        result.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        // Truncate to the `l` nearest
        result.truncate(l);

        result
    }

    /// Prune nodes in order to maintain edge diversity.
    pub fn prune(&mut self, mut candidates: Vec<(Dist<f32>, u32)>, a: f32) -> [u32; R] {
        let mut new_edge_nodes = [0; R]; // or [u32::MAX; R]

        // If you want to ensure we pick from the nearest to farthest:
        candidates.sort_by(|(dist_a, _), (dist_b, _)| dist_a.partial_cmp(dist_b).unwrap());

        // If candidates are already fewer than R, just fill and return.
        if candidates.len() <= R {
            for (i, (_, node_index)) in candidates.into_iter().enumerate() {
                new_edge_nodes[i] = node_index;
            }
            return new_edge_nodes;
        }

        let mut candidates: Vec<(Dist<f32>, P, u32)> = candidates
            .into_iter()
            .map(|(dist, node_index)| {
                let node = self
                    .storage
                    .get(&node_index)
                    .expect("Expected an active node");
                let point = node.point;
                (dist, point, node_index)
            })
            .collect();

        let mut edge_count = 0;
        let mut i = 0; // Current index of the candidate we're selecting

        // We keep going while we haven't selected R edges and still have candidates.
        while i < candidates.len() && edge_count < R {
            // Take the candidate at position `i` (assumed "nearest" if sorted).
            let (_, pa_point, pa) = candidates[i].clone();
            new_edge_nodes[edge_count] = pa;
            edge_count += 1;

            // Now we remove from the *remaining* candidates any that fail:
            //     α·d(p*, p') > d(p, p')
            // We'll do it in-place using `swap_remove`.
            let mut j = i + 1;
            while j < candidates.len() {
                let (dist_xp_pd, pd_point, _pd) = candidates[j].clone();
                let dist_pa_pd = pa_point.distance(&pd_point);

                // If this candidate *does not* meet the condition, we remove it.
                if a * dist_pa_pd <= dist_xp_pd.0 {
                    // Remove by swapping with the last element in O(1)
                    candidates.swap_remove(j);
                    // Don't advance j, because we need to check the swapped-in item.
                } else {
                    // It's good, keep it, move on
                    j += 1;
                }
            }

            // Move to the next candidate in the (now possibly shortened) list
            i += 1;
        }

        new_edge_nodes
    }

    pub fn insert(&mut self, point: P, a: f32, l: usize, rng: &mut T) -> u32 {
        // Search the new point and use route nodes as candidates of its edges
        let candidates = self.search(&point, l, rng);
        // display_instruction_counter("search");
        let edges = Self::prune(self, candidates, a);
        display_instruction_counter("prune new edges");

        // Add new node to the storage
        let new_node_index = self.storage.alloc();
        self.storage.set(new_node_index, Node { point, edges });

        // Update graph edges: add the new node to its edge node's edges
        let edges = edges
            .to_vec()
            .into_iter()
            .sorted()
            .dedup()
            .collect::<Vec<_>>();
        debug!("node {new_node_index}: {:?}", edges);
        // display_instruction_counter("start add_edge");
        for edge_node_index in edges {
            // If candidate list has delete nodes, uses undeleted edges of delete node instead.
            self.add_edge(&edge_node_index, new_node_index);
            // display_instruction_counter(&format!("add_edge,{edge_node_index}"));
        }

        debug!("\n");

        new_node_index
    }

    pub async fn delete<F, Fut>(&mut self, a: f32, commit: F) -> Vec<u32>
        where
            F: Fn() -> Fut,
            Fut: Future<Output = ()>,
    {
        let removed_indices: Vec<u32> = self
            .storage
            .clear_removed_indices()
            .into_iter()
            .sorted()
            .dedup()
            .collect();

        // Get all nodes that as removed indies in its edges
        let target_nodes: Vec<(u32, Node<P, R>)> = removed_indices
            .iter()
            .map(|removed_index| self.storage.backlink(*removed_index))
            .flatten()
            .sorted()
            .dedup()
            .filter_map(|p_index| match self.storage.get(&p_index) {
                Some(p_node) => Some((p_index, p_node)),
                None => None,
            }).collect();
        
        // Iterate all target nodes
        for (p_index, p_node) in target_nodes {

            // Candidates excluding removed indiecs
            let candidates: Vec<_> = p_node
                .edges()
                .into_iter()
                .filter_map(|edge_node_index| {
                    if removed_indices.contains(&edge_node_index) {
                        // Use its edges instead of this removed indices
                        match self.storage.get(&edge_node_index) {
                            Some(v_node) => Some(v_node.edges()),
                            None => None,
                        }
                    } else {
                        Some(vec![edge_node_index])
                    }
                })
                .flatten()
                .sorted()
                .dedup()
                .filter_map(|candidate_node_index| {
                    if removed_indices.contains(&candidate_node_index) {
                        return None
                    };
                    match self.storage.get(&candidate_node_index) {
                        Some(candidate_node) => {
                            let dist = Dist(candidate_node.point().distance(&p_node.point));
                            Some((dist, candidate_node_index))
                        },
                        None => None,
                    }
                })
                .collect();

            let new_edges = self.prune(candidates, a);

            // Update edge
            self.storage.set(p_index, Node {
                edges: new_edges,
                point: p_node.point(),
            });

            commit().await;
        }

        // Delete
        for removed_index in removed_indices.iter() {
            self.storage.delete(*removed_index);
        }

        removed_indices
    }

    pub fn remove(&mut self, index: u32) {
        self.storage.remove(index);
    }

    pub fn removed_indices(&self) -> Vec<u32> {
        self.storage.removed_indices()
    }

    // Add a new edge node to the node.
    fn add_edge(&mut self, target_node_index: &u32, new_edge_node_index: u32) {
        debug!("add_edge: {target_node_index}, new_edge_node_index: {new_edge_node_index}");
        let Some(target_node) = self.storage.get(target_node_index) else {
            panic!("")
        };
        let mut edges = target_node.edges();
        edges.push(new_edge_node_index);
        debug!("add_edge: {:?}", edges);

        let new_edges = if edges.len() > R {
            let candidates: Vec<_> = edges
                .into_iter()
                .map(|candidate_node_index| {
                    let Some(candidate_node) = self.storage.get(&candidate_node_index) else {
                        panic!("")
                    };
                    (
                        Dist(target_node.point().distance(&candidate_node.point())),
                        candidate_node_index,
                    )
                })
                .sorted() // candidates should be order for merging to list
                .dedup()
                .collect();

            self.prune(candidates, 2.0)
        } else {
            let mut new_edge_nodes = [0; R];
            for (i, edge_node_index) in edges.into_iter().enumerate() {
                new_edge_nodes[i] = edge_node_index;
            }

            new_edge_nodes
        };

        debug!(
            "add_edge: afger: node {target_node_index}: {:?}",
            new_edges
                .to_vec()
                .into_iter()
                .sorted()
                .dedup()
                .collect::<Vec<_>>()
        );

        self.storage.set(
            *target_node_index,
            Node {
                point: target_node.point(),
                edges: new_edges,
            },
        );
    }
}

fn display_instruction_counter(_name: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        use num_format::Locale;
        use num_format::ToFormattedString;

        let counter = ic_cdk::api::instruction_counter();
        ic_cdk::println!(
            "[Inst Counter] \"{_name}\": {}",
            counter.to_formatted_string(&Locale::en)
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::point::Point;

    use super::*;
    use itertools::Itertools;
    use rand::SeedableRng;
    use rand::{rngs::SmallRng, seq::SliceRandom};
    use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
    use rustc_hash::{FxHashMap, FxHashSet};

    const R: usize = 20;
    const L: usize = 125;

    fn normalize_to_unit_length(vector: Vec<f32>) -> Vec<f32> {
        let norm: f32 = vector.iter().map(|&v| v * v).sum::<f32>().sqrt(); // ユークリッドノルムを計算
        if norm.abs() < f32::EPSILON {
            // ノルムがゼロの場合（すべての要素が0）、そのまま返す
            return vector;
        }
        vector.into_iter().map(|v| v / norm).collect()
    }

    struct TestStorage {
        nodes: FxHashMap<u32, Node<Point, R>>,
        backlinks: FxHashMap<u32, FxHashSet<u32>>,
        index: u32,
        removed_indices: Vec<u32>,
    }

    impl StorageTrait<Point, R, SmallRng> for TestStorage {
        fn init() -> Self {
            Self {
                nodes: FxHashMap::default(),
                index: 0,
                backlinks: FxHashMap::default(),
                removed_indices: Vec::new(),
            }
        }

        fn get(&self, index: &u32) -> Option<Node<Point, R>> {
            self.nodes.get(index).cloned()
        }

        fn set(&mut self, index: u32, active_node: Node<Point, R>) -> Option<Node<Point, R>> {
            // Set rc of this node
            match self.backlinks.get(&index) {
                Some(_) => {}
                None => {
                    self.backlinks.insert(index, FxHashSet::default());
                }
            }
            //
            let old_node = self.nodes.insert(index, active_node.clone());
            let old_edges = match &old_node {
                Some(node) => node.edges(),
                None => {
                    vec![]
                }
            };
            // Delete backlinks // 前に持っていたedgeからのbacklinkを一度全て削除する。
            for edge_index in old_edges {
                let Some(set) = self.backlinks.get_mut(&edge_index) else {
                    panic!("")
                };
                set.remove(&index);
            }
            // Add backlinks
            for edge_index in active_node.edges.into_iter().sorted().dedup() {
                let Some(set) = self.backlinks.get_mut(&edge_index) else {
                    panic!("")
                };
                set.insert(index);
            }

            old_node
        }

        fn delete(&mut self, index: u32) {
            let Some(_node) = self.nodes.get_mut(&index) else {
                return;
            };

            todo!()
        }

        fn remove(&mut self, index: u32) {
            self.removed_indices.push(index);
        }

        fn removed_indices(&self) -> Vec<u32> {
            self.removed_indices.clone()
        }

        fn clear_removed_indices(&mut self) -> Vec<u32> {
            let list = self.removed_indices.clone();
            self.removed_indices = vec![];
            list
        }

        fn alloc(&mut self) -> u32 {
            self.index += 1;
            self.index
        }

        fn pick_random_active_node(&self, rng: &mut SmallRng) -> (u32, Node<Point, R>) {
            let keys: Vec<_> = self.nodes.keys().collect();

            loop {
                let Some(&random_key) = keys.get(rng.gen_range(0..keys.len())) else {
                    panic!("btree is empty")
                };

                let Some(active_node) = self.get(random_key) else {
                    continue;
                };

                return (*random_key, active_node);
            }
        }

        fn backlink(&self, _index: u32) -> Vec<u32> {
            todo!()
        }
    }

    const DIM: usize = 384;

    #[test]
    fn test_get() {
        let mut rng: SmallRng = SmallRng::from_entropy();
        let mut graph: Graph<Point, TestStorage, R, SmallRng> = Graph::init(DIM);

        env_logger::init();

        let iter_count = 500;
        let split_count = 0;

        // Generate random poins
        let test_points: Vec<(u32, Point)> = (0..iter_count)
            .into_iter()
            .map(|i| (i + 1, gen_point(&mut rng)))
            .collect();

        // Insert them into Vectune
        for (index, p) in test_points.iter() {
            let new_index = graph.insert(p.clone(), 2.0, L, &mut rng);
            assert_eq!(*index, new_index);
        }

        // Recall rate
        let hit_counts: f32 = test_points[split_count as usize..]
            .par_iter()
            .map_init(
                || SmallRng::from_entropy(),
                |rng, (_, p)| {
                    let p = p.clone();
                    let ground_truth: Vec<(Dist<f32>, u32)> = test_points
                        .iter()
                        .map(|(i, q)| (Dist(p.distance(q)), *i))
                        .sorted()
                        .collect();

                    let result = graph.search(&p, L, rng);

                    // println!("{:?}, {:?}", ground_truth.split(6), result[0..6]);
                    let _g_list: Vec<_> = ground_truth[0..5]
                        .into_iter()
                        .map(|(_, index)| index)
                        .collect();
                    let _r_list: Vec<_> =
                        result[0..5].into_iter().map(|(_, index)| index).collect();

                    let set1: FxHashSet<_> = ground_truth[0..5]
                        .into_iter()
                        .map(|(_, index)| index)
                        .collect();
                    let set2: FxHashSet<_> =
                        result[0..5].into_iter().map(|(_, index)| index).collect();

                    let intersection_count = set1.intersection(&set2).count();

                    intersection_count as f32
                },
            )
            .sum();

        println!("{:?}", hit_counts / ((5 * iter_count) as f32));
    }

    #[test]
    fn test_remove() {
        let mut rng: SmallRng = SmallRng::from_entropy();
        let mut graph: Graph<Point, TestStorage, R, SmallRng> = Graph::init(DIM);

        env_logger::init();

        let iter_count = 100;

        // Generate random poins
        let mut test_points: Vec<(u32, Point)> = (0..iter_count)
            .into_iter()
            .map(|i| (i + 1, gen_point(&mut rng)))
            .collect();

        // Insert them into Vectune
        for (index, p) in test_points.iter() {
            let new_index = graph.insert(p.clone(), 2.0, L, &mut rng);
            assert_eq!(*index, new_index);
        }

        // Remove some indices
        test_points.shuffle(&mut rng);
        let removed_indices: Vec<u32> = test_points[0..10]
            .to_vec()
            .into_iter()
            .map(|(i, _)| {
                graph.remove(i);
                i
            })
            .collect();

        // Assert graph.removed_indieces equals to the removed_indices
        let graph_removed_indices = graph.removed_indices();
        assert_eq!(graph_removed_indices.len(), removed_indices.len());
        graph_removed_indices
            .into_par_iter()
            .for_each(|i| assert!(removed_indices.contains(&i)));

        // Assert result does not include removed indices
        test_points.par_iter().for_each_init(
            || SmallRng::from_entropy(),
            |rng, (_, p)| {
                let result = graph.search(p, L, rng);
                result
                    .into_iter()
                    .for_each(|(_, i)| assert!(!removed_indices.contains(&i)));
            },
        );
    }

    fn gen_point(rng: &mut SmallRng) -> Point {
        let new_point: Vec<f32> = (0..DIM).map(|_| rng.gen_range(-1000.0..1000.0)).collect();
        let new_point = normalize_to_unit_length(new_point);
        Point::new(new_point.try_into().unwrap(), DIM)
    }

    #[test]
    fn test_backlinks() {
        // rcが、一致しているかをテストする。
        // edgeを変更した時に、rcが正しく増減するかをテストする。
    }
}
