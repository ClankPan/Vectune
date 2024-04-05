use std::time::Instant;

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::{rngs::SmallRng, Rng};

use rayon::prelude::*;
use std::sync::RwLock;
// use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

pub mod kmeans;
pub mod pq;

#[derive(Serialize, Deserialize, Clone)]
pub struct Builder {
    a: f32,
    r: usize,
    pub l: usize,
    seed: u64,
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
    pub fn build<P: Point, V: Clone>(self, points: Vec<P>, values: Vec<V>) -> FreshVamanaMap<P, V> {
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

        Self { ann, values }
    }
    pub fn search(&self, query_point: &P) -> Vec<(f32, V)> {
        let (results, _visited) = self.ann.greedy_search_v2(query_point, 30, self.ann.builder.l);
        // println!("\n\nvisited:   {:?}\n\n", _visited);
        results
            .into_iter()
            .map(|(dist, i)| (dist, self.values[i].clone()))
            .collect()
    }
}

#[derive(Serialize, Deserialize)]
struct Node<P> {
    n_out: RwLock<Vec<usize>>, // has pointer. ToDo: should use PQ to resuce memory accesses.
    n_in: Vec<usize>,
    p: P,
    id: usize,
}

#[derive(Serialize, Deserialize)]
pub struct FreshVamana<P> {
    nodes: Vec<Node<P>>,
    centroid: usize,
    pub builder: Builder,
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
        println!("rand init phase");
        let start_time = Instant::now();
        let mut ann = FreshVamana::<P>::random_graph_init(points, builder, &mut rng);
        println!(
            "\nrand init time: {:?}",
            Instant::now().duration_since(start_time)
        );

        // Prune Edges
        FreshVamana::<P>::indexing(&mut ann, &mut rng);

        println!(
            "\ntotal indexing time: {:?}",
            Instant::now().duration_since(start_time)
        );

        ann
    }

    fn random_graph_init(points: Vec<P>, builder: Builder, rng: &mut SmallRng) -> Self {
        if points.is_empty() {
            return Self {
                nodes: Vec::new(),
                centroid: usize::MAX,
                builder,
                cemetery: Vec::new(),
                empties: Vec::new(),
            };
        }

        assert!(points.len() < u32::MAX as usize);
        let points_len = points.len();

        /* Find Centroid */
        let mut average_point: Vec<f32> = vec![0.0; P::dim() as usize];
        for p in &points {
            average_point = p
                .to_f32_vec()
                .iter()
                .zip(average_point.iter())
                .map(|(x, y)| x + y)
                .collect();
        }
        let average_point = P::from_f32_vec(
            average_point
                .into_iter()
                .map(|v| v / points_len as f32)
                .collect(),
        );
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
        let nodes: Vec<Node<P>> = points
            .into_iter()
            .enumerate()
            .map(|(id, p)| Node {
                n_out: RwLock::new(Vec::new()),
                n_in: Vec::new(),
                p,
                id,
            })
            .collect();

        let mut working: Vec<usize> = (0..nodes.len()).collect();
        let node_len = nodes.len();

        for node_i in 0..node_len {
            let mut n_out_cout = 0;
            while n_out_cout < builder.r {
                let working_i = rng.gen_range(0..working.len());
                if working_i == node_i {
                    continue;
                } else {
                    n_out_cout += 1;
                }
                let out_node_i = working[working_i];
                // insert_id(node_i, &mut nodes[out_node_i].n_in); // ToDo: refactor , use self.make_edge()
                insert_id(out_node_i, &mut nodes[node_i].n_out.write().unwrap());

                // Since prevents the creation of nodes that are not referenced by anyone during initialization,
                // ensure that all input edges are R nodes
                if nodes[out_node_i].n_in.len() == builder.r {
                    working.remove(working_i);
                }
            }
        }

        Self {
            nodes,
            centroid,
            builder,
            cemetery: Vec::new(),
            empties: Vec::new(),
        }
    }

    fn indexing(ann: &mut FreshVamana<P>, rng: &mut SmallRng) {
        let node_len = ann.nodes.len();
        let mut shuffled: Vec<usize> = (0..node_len).collect();
        shuffled.shuffle(rng);

        // for 1 ≤ i ≤ n do
        shuffled.into_iter().enumerate().for_each(|(count, i)| {
            if count % 10000 == 0 {
                println!("id : {}\t/{}", count, ann.nodes.len());
            }

            // let [L; V] ← GreedySearch(s, xσ(i), 1, L)
            let (_, visited) = ann.greedy_search_v2(&ann.nodes[i].p, 1, ann.builder.l);

            // V ← (V ∪ Nout(p)) \ {p}
            let prev_n_out = ann.nodes[i].n_out.read().unwrap().clone();
            let mut candidates = visited;
            for out_i in &prev_n_out {
                if !is_contained_in(out_i, &candidates) {
                    // let dist = self.node_distance(xp, out_i);
                    let dist = ann.nodes[i].p.distance(&ann.nodes[*out_i].p);
                    insert_dist((dist, *out_i), &mut candidates)
                }
            }

            // run RobustPrune(σ(i), V, α, R) to update out-neighbors of σ(i)
            let new_n_out = ann.prune(&mut candidates);
            // robust prune実行中に追加されたnoutを出す。
            {
                let mut current_n_out = ann.nodes[i].n_out.write().unwrap();
                let new_added_ids = diff_ids(&current_n_out, &prev_n_out);
                *current_n_out = new_n_out.clone();
                for out_i in new_added_ids {
                    insert_id(out_i, &mut current_n_out);
                }
            } // unlock the write lock

            // for all points j in Nout(σ(i)) do
            for j in new_n_out {
                if ann.nodes[j].n_out.read().unwrap().contains(&i) {
                    continue;
                } else {
                    // Todo : refactor, self.make_edge　or union. above ann.nodes[j].n_out.contains(&i) not necessary if use union
                    insert_id(i, &mut ann.nodes[j].n_out.write().unwrap());
                    // insert_id(j, &mut ann.nodes[i].n_in);
                }

                // // if |Nout(j) ∪ {σ(i)}| > R then run RobustPrune(j, Nout(j) ∪ {σ(i)}, α, R) to update out-neighbors of j
                // let prev_n_out = ann.nodes[j].n_out.read().unwrap().clone();
                // let j_point =  &ann.nodes[j].p;
                // if prev_n_out.len() > ann.builder.r {
                //   // robust_prune requires (dist(xp, p'), index)
                //   let mut candidates: Vec<(f32, usize)> = prev_n_out.iter()
                //     .map(|out_i|
                //       // (ann.nodes[out_i].p.distance(j_point), out_i)
                //       (ann.nodes[*out_i].p.distance(j_point), *out_i)
                //     ).collect();

                //   let new_n_out = ann.prune(&mut candidates);

                //   {
                //     let mut current_n_out = ann.nodes[j].n_out.write().unwrap();
                //     let new_added_ids = diff_ids(&current_n_out, &prev_n_out);
                //     *current_n_out = new_n_out.clone();
                //     for out_i in new_added_ids {
                //       insert_id(out_i, &mut current_n_out);
                //     }
                //   } // unlock the write lock
                // }
            }
        });
    }

    fn prune(&self, candidates: &mut Vec<(f32, usize)>) -> Vec<usize> {
        let mut new_n_out = vec![];

        while let Some((first, rest)) = candidates.split_first() {
            let (_, pa) = *first; // pa is p asterisk (p*), which is nearest point to p in this loop
            new_n_out.push(pa);

            if new_n_out.len() == self.builder.r {
                break;
            }
            *candidates = rest.to_vec();

            // if α · d(p*, p') <= d(p, p') then remove p' from v
            candidates.retain(|&(dist_xp_pd, pd)| {
                let pa_point = &self.nodes[pa].p;
                let pd_point = &self.nodes[pd].p;
                let dist_pa_pd = pa_point.distance(pd_point);

                self.builder.a * dist_pa_pd > dist_xp_pd
            })
        }

        new_n_out
    }

    fn greedy_search_v2(
        &self,
        query_point: &P,
        k: usize,
        l: usize,
    ) -> (Vec<(f32, usize)>, Vec<(f32, usize)>) {
        // k-anns, visited
        assert!(l >= k);
        let s = self.centroid;
        let mut visited: Vec<(f32, usize)> = Vec::new();
        let mut touched: Vec<usize> = Vec::with_capacity(self.builder.l*2);

        // let mut list: Vec<(f32, usize, bool)> = self.nodes[s]
        //     .n_out
        //     .read()
        //     .unwrap()
        //     .clone()
        //     .into_iter()
        //     .map(|out_i|{
        //         // insert_id(out_i, &mut touched);
        //         (query_point.distance(&self.nodes[out_i].p), out_i, false)
        //     })
        //     .collect();
        // sort_list_by_dist(&mut list);
        // list.truncate(self.builder.l);

        let mut list: Vec<(f32, usize, bool)> = vec![(query_point.distance(&self.nodes[s].p), s, true)];
        let mut working = Some((0, list[0]));
        visited.push((list[0].0, list[0].1));
        insert_id(list[0].1, &mut touched);

        while let Some((list_i, (nearest_dist, nearest_i, _))) = working {

            visited.push((nearest_dist, nearest_i));
            insert_id(nearest_i, &mut touched);
            list[list_i].2 = true;

            let mut nouts: Vec<(f32, usize, bool)> = self.nodes[nearest_i]
                .n_out
                .read()
                .unwrap()
                .clone()
                .iter()
                .filter_map(|out_i| {
                    if list.iter().find(|(_, id, _)| id == out_i).is_some() {
                        return None
                    }
                    match touched.binary_search(out_i) {
                        Ok(_) => return None,
                        Err(_index) => {
                            // touched.insert(index, *out_i);
                            return Some((query_point.distance(&self.nodes[*out_i].p), *out_i, false))
                        }
                    }
                })
                .collect();
            sort_list_by_dist(&mut nouts);

            // ここを短くする。
            list.extend(nouts);
            sort_list_by_dist(&mut list);
            list.truncate(self.builder.l);

            

            working = list.clone().into_iter().enumerate().find(|(_, (_, _, is_visited))| !is_visited);

            // println!("working: {:?}\n\n", working);
            // println!("list: {:?}\n\n", list);
            // println!("visited: {:?}\n\n", visited);
            // println!("touched: {:?}\n\n", touched);
        }

        let mut k_anns = list
            .into_iter()
            .map(|(dist, id, _)| (dist, id))
            .collect::<Vec<(f32, usize)>>();

        k_anns.truncate(k);

        sort_list_by_dist_v1(&mut visited);

        (k_anns, visited)

    }

    fn greedy_search(&self, xq: &P, k: usize, l: usize) -> (Vec<(f32, usize)>, Vec<(f32, usize)>) {
        // k-anns, visited
        assert!(l >= k);
        let s = self.centroid;
        let mut visited: Vec<(f32, usize)> = Vec::new();
        let mut list: Vec<(f32, usize)> = vec![(self.nodes[s].p.distance(xq), s)];

        // `working` is a list of unexplored candidates
        let mut working = list.clone(); // Because list\visited == list at beginning

        fn contains_or_insert(value: (f32, usize), vec: &mut Vec<(f32, usize)>) -> bool {
            match find_index_by_id(value.1, vec)
            {
                Ok(_) => {
                    true
                },
                Err(index) => {
                    vec.insert(index, value);
                    false
                }
            }
        }

        fn contains_and_remove(id: usize, vec: &mut Vec<(f32, usize)>) -> bool {
            match find_index_by_id(id, vec)
            {
                Ok(index) => {
                    vec.remove(index);
                    true
                },
                Err(_) => {
                    false
                }
            }
        }

        fn find_index_by_id(id: usize, vec: &Vec<(f32, usize)>) -> Result<usize, usize> {
            vec.binary_search_by(|probe| {
                probe
                    .1
                    .cmp(&id)
            })
        }

        fn contains_id(id: usize, vec: &Vec<(f32, usize)>) -> bool {
            match find_index_by_id(id, vec)
            {
                Ok(_) => {
                    true
                },
                Err(_) => {
                    false
                }
            }
        }

        fn diff_by_ids(a: &Vec<(f32, usize)>, b: &Vec<(f32, usize)>) -> Vec<(f32, usize)> {
            let mut result = Vec::new();
            let mut a_idx = 0;
            let mut b_idx = 0;

            while a_idx < a.len() && b_idx < b.len() {
                if a[a_idx].1 == b[b_idx].1 {
                    a_idx += 1; // Skip common elements
                    b_idx += 1;
                } else if a[a_idx].1 < b[b_idx].1 {
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

        // let p∗ ← arg minp∈L\V ||xp − xq||
        while let Some(nearest) = working
            .iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
        {
            // If the nearest node is already visited
            if contains_or_insert(nearest, &mut visited) {
                continue;
            }

            // If the node is marked as grave, remove from result list. But Its neighboring nodes are explored.
            if self.cemetery.contains(&nearest.1) {
                contains_and_remove(nearest.1, &mut list);
            }

            // update L ← L ∪ Nout(p∗) and V ← V ∪ {p∗}
            let nearest_n_out = &self.nodes[nearest.1].n_out.read().unwrap().clone();
            for out_i in nearest_n_out {
                let out_i_point = &self.nodes[*out_i].p;

                // if is_contained_in(out_i, &list) || is_contained_in(out_i, &visited) {
                //     // Should check visited as grave point is in visited but not in list.
                //     continue;
                // }
                if contains_id(*out_i, &list) || contains_id(*out_i, &visited) {
                    continue
                }

                let dist = xq.distance(out_i_point);
                // list.push((dist, node_i));
                // insert_dist((dist, *out_i), &mut list);
                contains_or_insert((dist, *out_i), &mut list);
            }

            if list.len() > l {
                // sort_by_dist_and_resize(&mut list, l)
                sort_list_by_dist_v1(&mut list);
                list.truncate(l);
                sort_list_by_id(&mut list)
            }

            working = diff_by_ids(&list, &visited);
        }

        sort_by_dist_and_resize(&mut list, k);
        let k_anns = list;

        (k_anns, visited)
    }
}

fn set_diff(a: Vec<(f32, usize)>, b: &Vec<(f32, usize)>) -> Vec<(f32, usize)> {
    a.into_iter()
        .filter(|(_, p)| !is_contained_in(p, b))
        .collect()
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

fn sort_list_by_dist(list: &mut Vec<(f32, usize, bool)>) {
    list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less));
}

fn sort_list_by_dist_v1(list: &mut Vec<(f32, usize)>) {
    list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less));
}

fn sort_list_by_id(list: &mut Vec<(f32, usize)>) {
    list.sort_by(|a, b| a.1.cmp(&b.1));
}

// fn find_nearest(c: &mut Vec<(f32, usize)>) -> (f32, usize) {
//     sort_list_by_dist(c); // ToDo: Ensure that the arugment list is already sorted.
//     c[0]
// }

fn sort_by_dist_and_resize(list: &mut Vec<(f32, usize)>, size: usize) {
    list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less));
    list.truncate(size)
}

fn is_contained_in(i: &usize, vec: &Vec<(f32, usize)>) -> bool {
    !vec.iter()
        .filter(|(_, id)| *id == *i)
        .collect::<Vec<&(f32, usize)>>()
        .is_empty()
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
        }
        Err(_) => {
            // If the value of f32 is not found, nothing is done.
        }
    }
}

fn insert_id(value: usize, vec: &mut Vec<usize>) {
    match vec.binary_search(&value) {
        Ok(_index) => { // If already exsits
        }
        Err(index) => {
            vec.insert(index, value);
        }
    }
}

fn insert_dist(value: (f32, usize), vec: &mut Vec<(f32, usize)>) {
    match vec.binary_search_by(|probe| {
        probe
            .0
            .partial_cmp(&value.0)
            .unwrap_or(std::cmp::Ordering::Less)
    }) {
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
        }
        Err(index) => {
            vec.insert(index, value);
        }
    };
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

    #[derive(Clone, Debug)]
    struct Point(Vec<u32>);
    impl VPoint for Point {
        fn distance(&self, other: &Self) -> f32 {
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(a, b)| (*a as f32 - *b as f32).powi(2))
                .sum::<f32>()
                .sqrt()
        }
        fn dim() -> u32 {
            12
        }
        fn to_f32_vec(&self) -> Vec<f32> {
            self.0.iter().map(|v| *v as f32).collect()
        }
        fn from_f32_vec(a: Vec<f32>) -> Self {
            Point(a.into_iter().map(|v| v as u32).collect())
        }
    }

    #[test]
    fn test_random_init_v2() {
        let builder = Builder::default();
        // let seed = builder.seed;
        let seed: u64 = 11923543545843533243;
        let mut rng = SmallRng::seed_from_u64(seed);
        println!("seed: {}", seed);

        let mut i = 0;

        let points: Vec<Point> = (0..100)
            .map(|_| {
                let a = i;
                i += 1;
                Point(vec![a; Point::dim() as usize])
            })
            .collect();

        let point_len = points.len();

        let ann: FreshVamana<Point> = FreshVamana::random_graph_init(points, builder, &mut rng);

        for node_i in 0..point_len {
            for out_i in ann.nodes[node_i].n_out.read().unwrap().iter() {
                assert!(ann.nodes[*out_i].n_in.contains(&node_i))
            }
            for in_i in &ann.nodes[node_i].n_in {
                assert!(ann.nodes[*in_i].n_out.read().unwrap().contains(&node_i))
            }
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

        let points: Vec<Point> = (0..100)
            .map(|_| {
                let a = i;
                i += 1;
                Point(vec![a; Point::dim() as usize])
            })
            .collect();
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

        let points: Vec<Point> = (0..1000)
            .map(|_| {
                let a = i;
                i += 1;
                Point(vec![a; Point::dim() as usize])
            })
            .collect();

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
    fn greedy_search() {
        let mut builder = Builder::default();
        builder.set_l(30);
        println!("seed: {}", builder.seed);
        // let seed = builder.seed;
        let seed: u64 = 17674802184506369839;
        let mut rng = SmallRng::seed_from_u64(seed);
        let l = builder.l;

        let mut i = 0;

        let points: Vec<Point> = (0..500)
            .map(|_| {
                let a = i;
                i += 1;
                Point(vec![a; Point::dim() as usize])
            })
            .collect();

        let ann: FreshVamana<Point> = FreshVamana::random_graph_init(points, builder, &mut rng);
        let xq = Point(vec![0; Point::dim() as usize]);
        let k = 10;
        let (k_anns, _visited) = ann.greedy_search_v2(&xq, k, l);

        println!("k_anns: {:?}", k_anns);

        for i in 0..10 {
            assert_eq!(k_anns[i].1, i);
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
        let mut a = vec![
            (0.2, 2, false),
            (0.1, 1, false),
            (0.3, 3, false),
            (0.0, 0, false),
        ];
        sort_list_by_dist(&mut a);
        assert_eq!(
            a,
            vec![
                (0.0, 0, false),
                (0.1, 1, false),
                (0.2, 2, false),
                (0.3, 3, false)
            ]
        )
    }

    // #[test]
    // fn test_find_nearest() {
    //     let mut a = vec![(0.2, 2), (0.1, 1), (0.3, 3), (0.0, 0)];
    //     assert_eq!(find_nearest(&mut a), (0.0, 0));
    // }

    #[test]
    fn test_sort_and_resize() {
        let mut a = vec![(0.2, 2), (0.1, 1), (0.3, 3), (0.0, 0)];
        sort_by_dist_and_resize(&mut a, 2);
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
        let mut a = vec![0, 1, 3, 4];
        insert_id(2, &mut a);
        insert_id(2, &mut a);
        assert_eq!(a, vec![0, 1, 2, 3, 4]);

        let mut a = vec![1, 3, 4];
        insert_id(0, &mut a);
        assert_eq!(a, vec![0, 1, 3, 4])
    }

    #[test]
    fn test_insert_dist() {
        let mut a = vec![(0.0, 0), (0.1, 1), (0.3, 3)];
        insert_dist((0.2, 2), &mut a);
        insert_dist((0.2, 2), &mut a);
        assert_eq!(a, vec![(0.0, 0), (0.1, 1), (0.2, 2), (0.3, 3)]);

        let mut a = vec![
            (0.0, 1),
            (1.7320508, 2),
            (3.4641016, 3),
            (5.196152, 4),
            (6.928203, 5),
            (8.6602545, 6),
            (12.124355, 8),
            (13.856406, 9),
            (15.588457, 10),
            (17.320509, 11),
            (19.052559, 12),
            (20.784609, 13),
            (22.51666, 14),
            (24.24871, 15),
            (27.712812, 17),
            (862.5613, 499),
        ];
        insert_dist((1.7320508, 0), &mut a);
        assert_eq!(
            a,
            vec![
                (0.0, 1),
                (1.7320508, 0),
                (1.7320508, 2),
                (3.4641016, 3),
                (5.196152, 4),
                (6.928203, 5),
                (8.6602545, 6),
                (12.124355, 8),
                (13.856406, 9),
                (15.588457, 10),
                (17.320509, 11),
                (19.052559, 12),
                (20.784609, 13),
                (22.51666, 14),
                (24.24871, 15),
                (27.712812, 17),
                (862.5613, 499)
            ]
        )
    }

    // #[test]
    // fn test_diff_ids() {
    //     let a = vec![0, 1, 3, 4];
    //     let b = vec![0, 4];
    //     let c = diff_ids(&a, &b);
    //     assert_eq!(c, vec![1, 3]);

    //     let b = vec![0, 4, 5];
    //     let c = diff_ids(&a, &b);
    //     assert_eq!(c, vec![1, 3]);

    //     let b = vec![0, 1, 3, 4];
    //     let c = diff_ids(&a, &b);
    //     assert_eq!(c, vec![]);

    //     let b = vec![];
    //     let c = diff_ids(&a, &b);
    //     assert_eq!(c, a);
    // }
}
