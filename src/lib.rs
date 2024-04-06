use rustc_hash::FxHashSet;
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::{rngs::SmallRng, Rng};

use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

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
    pub fn build<P: Point, V: Clone>(self, points: Vec<P>) -> (Vec<(P, Vec<usize>)>, usize) {
        let ann = Vamana::new(points, self);

        let nodes = ann
            .nodes
            .into_iter()
            .map(|node| (node.p, node.n_out.into_inner()))
            .collect();
        let s = ann.centroid;

        (nodes, s)
    }
}

pub trait Graph<P> {
    fn alloc(&mut self, point: P) -> usize;
    fn free(&mut self, id: &usize);
    fn cemetery(&self) -> Vec<usize>;
    fn clear_cemetery(&mut self);
    fn backlink(&self, id: &usize) -> Vec<usize>;
    fn get(&self, id: &usize) -> (P, Vec<usize>);
    fn size_l(&self) -> usize;
    fn size_r(&self) -> usize;
    fn size_a(&self) -> f32;
    fn start_id(&self) -> usize;
    fn overwirte_out_edges(&mut self, id: &usize, edges: Vec<usize>); // backlinkを処理する必要がある。
}

pub fn search<P, G>(
    graph: &mut G,
    query_point: &P,
    k: usize,
) -> (Vec<(f32, usize)>, Vec<(f32, usize)>)
where
    P: Point,
    G: Graph<P>,
{
    // k-anns, visited
    let builder_l = graph.size_l();
    assert!(builder_l >= k);

    let mut visited: Vec<(f32, usize)> = Vec::with_capacity(builder_l * 2);
    let mut touched = FxHashSet::default();
    touched.reserve(builder_l * 100);

    let mut list: Vec<(f32, usize, bool)> = Vec::with_capacity(builder_l);
    let s = graph.start_id();
    let (s_point, _) = graph.get(&s);
    // list.push((query_point.distance(&s_point), s, true));
    list.push((query_point.distance(&s_point), s, true));
    let mut working = Some(list[0]);
    visited.push((list[0].0, list[0].1));
    touched.insert(list[0].1);

    while let Some((_, nearest_i, _)) = working {
        let (_, nearest_n_out) = graph.get(&nearest_i);
        let mut nouts: Vec<(f32, usize, bool)> = Vec::with_capacity(nearest_n_out.len());
        for out_i in nearest_n_out {
            if !touched.contains(&out_i) {
                touched.insert(out_i);
                let (out_point, _) = graph.get(&out_i);
                nouts.push((query_point.distance(&out_point), out_i, false))
            }
        }

        sort_list_by_dist(&mut nouts);

        let mut new_list = Vec::with_capacity(builder_l);
        let mut new_list_idx = 0;

        let mut l_idx = 0; // Index for list
        let mut n_idx = 0; // Index for dists

        working = None;

        while new_list_idx < builder_l {
            let mut new_min = if l_idx >= list.len() && n_idx >= nouts.len() {
                break;
            } else if l_idx >= list.len() {
                let new_min = nouts[n_idx];
                n_idx += 1;
                new_min
            } else if n_idx >= nouts.len() {
                let new_min = list[l_idx];
                l_idx += 1;
                new_min
            } else {
                let l_min = list[l_idx];
                let n_min = nouts[n_idx];

                if l_min.0 <= n_min.0 {
                    l_idx += 1;
                    l_min
                } else {
                    n_idx += 1;
                    n_min
                }
            };

            let is_not_visited = !new_min.2;

            if working.is_none() && is_not_visited {
                new_min.2 = true; // Mark as visited
                working = Some(new_min);
                visited.push((new_min.0, new_min.1));
            }

            // Deleted and visited nodes are not added.
            // Even if it is deleted, its neighboring nodes are included in the search candidates.
            if !graph.cemetery().contains(&new_min.1) || is_not_visited {
                new_list.push(new_min);
                new_list_idx += 1;
            }
        }

        list = new_list;
    }

    let mut k_anns = list
        .into_iter()
        .map(|(dist, id, _)| (dist, id))
        .collect::<Vec<(f32, usize)>>();
    k_anns.truncate(k);

    sort_list_by_dist_v1(&mut visited);

    (k_anns, visited)
}

pub fn insert<P, G>(graph: &mut G, new_p: P)
where
    P: Point,
    G: Graph<P>,
{
    let new_id = graph.alloc(new_p.clone());
    let r = graph.size_r();
    let a = graph.size_a(); 

    // [L, V] ← GreedySearch(𝑠, 𝑝, 1, 𝐿)
    let (_list, mut visited) = search(graph, &new_p, 1);
    // 𝑁out(𝑝) ← RobustPrune(𝑝, V, 𝛼, 𝑅) (Algorithm 3)
    let n_out = prune(
        |id| graph.get(id),
        &mut visited,
        &r,
        &a,
    );

    // foreach 𝑗 ∈ 𝑁out(𝑝) do
    for j in &n_out {
        // |𝑁out(𝑗) ∪ {𝑝}|
        let (j_point, mut j_n_out) = graph.get(j);
        j_n_out.push(new_id);
        j_n_out.sort();
        j_n_out.dedup();
        // if |𝑁out(𝑗) ∪ {𝑝}| > 𝑅 then
        if j_n_out.len() > r {
            // 𝑁out(𝑗) ← RobustPrune(𝑗, 𝑁out(𝑗) ∪ {𝑝}, 𝛼, 𝑅)
            let mut j_n_out_with_dist = j_n_out
                .iter()
                .map(|j_out_idx| (j_point.distance(&new_p), *j_out_idx))
                .collect::<Vec<(f32, usize)>>();
            sort_list_by_dist_v1(&mut j_n_out_with_dist);
            j_n_out = prune(
                |id| graph.get(id),
                &mut j_n_out_with_dist,
                &r,
                &a,
            );
        }
        graph.overwirte_out_edges(j, j_n_out);
    }

}

pub fn delete<P, G>(mut graph: G)
where
    P: Point,
    G: Graph<P>,
{
    /* 𝑝 ∈ 𝑃 \ 𝐿𝐷 s.t. 𝑁out(𝑝) ∩ 𝐿𝐷 ≠ ∅ */

    // Note: 𝐿𝐷 is Deleted List
    let mut ps = Vec::new();

    // s.t. 𝑁out(𝑝) ∩ 𝐿𝐷 ≠ ∅
    let mut cemetery = graph.cemetery();
    cemetery.sort();
    cemetery.dedup();

    for grave_i in &cemetery {
        ps.extend(graph.backlink(grave_i))
    }
    ps.sort();
    ps.dedup();

    // 𝑝 ∈ 𝑃 \ 𝐿𝐷
    ps = diff_ids(&ps, &cemetery);

    for p in ps {
        // D ← 𝑁out(𝑝) ∩ 𝐿𝐷
        let (_, p_n_out) = graph.get(&p);
        let d = intersect_ids(&p_n_out, &cemetery);
        // C ← 𝑁out(𝑝) \ D //initialize candidate list
        let mut c = diff_ids(&p_n_out, &d);

        // foreach 𝑣 ∈ D do
        for u in &d {
            // C ← C ∪ 𝑁out(𝑣)
            // c = union_ids(&c, &self.nodes[*u].n_out);
            let (_, u_n_out) = graph.get(u);
            c.extend(u_n_out);
            c.sort();
            c.dedup();
        }

        // C ← C \ D
        /*
        Note:
            Since D's Nout may contain LD, Why pull the D instead of the LD?
            I implemented it as shown and it hit data that should have been erased, so I'll fix it to pull LD.
        */
        c = diff_ids(&c, &cemetery);

        // 𝑁out(𝑝) ← RobustPrune(𝑝, C, 𝛼, 𝑅)
        //   let (p_point, _) = self.nodes[p].p.clone();
        let (p_point, _) = graph.get(&p);
        let mut c_with_dist: Vec<(f32, usize)> = c
            .into_iter()
            .map(|id| (p_point.distance(&graph.get(&id).0), id))
            .collect();

        sort_list_by_dist_v1(&mut c_with_dist);

        /*
        Note:
            Before call robust_prune, clean Nout(p) because robust_prune takes union v and Nout(p) inside.
            It may ontain deleted points.
            The original paper does not explicitly state in Algorithm 4.
        */
        let new_edges = prune(
            |id| graph.get(id),
            &mut c_with_dist,
            &graph.size_r(),
            &graph.size_a(),
        );
        graph.overwirte_out_edges(&p, new_edges);
    }

    for grave_i in &cemetery {
        graph.overwirte_out_edges(grave_i, vec![]); // Backlinks are not defined in the original algorithm but should be deleted here.
    }

    for grave_i in &cemetery {
        graph.free(grave_i)
    }

    graph.clear_cemetery();
}

fn prune<P, F>(
    get: F,
    candidates: &mut Vec<(f32, usize)>,
    builder_r: &usize,
    builder_a: &f32,
) -> Vec<usize>
where
    P: Point,
    F: Fn(&usize) -> (P, Vec<usize>),
{
    let mut new_n_out = vec![];

    while let Some((first, rest)) = candidates.split_first() {
        let (_, pa) = *first; // pa is p asterisk (p*), which is nearest point to p in this loop
        new_n_out.push(pa);

        if new_n_out.len() == *builder_r {
            break;
        }
        *candidates = rest.to_vec();

        // if α · d(p*, p') <= d(p, p') then remove p' from v
        candidates.retain(|&(dist_xp_pd, pd)| {
            // let pa_point = &self.nodes[pa].p;
            // let pd_point = &self.nodes[pd].p;
            let (pa_point, _) = get(&pa);
            let (pd_point, _) = get(&pd);
            let dist_pa_pd = pa_point.distance(&pd_point);

            builder_a * dist_pa_pd > dist_xp_pd
        })
    }

    new_n_out
}

struct Node<P> {
    n_out: RwLock<Vec<usize>>,
    p: P,
}

pub struct Vamana<P> {
    nodes: Vec<Node<P>>,
    centroid: usize,
    builder: Builder,
}

impl<P> Vamana<P>
where
    P: Point,
{
    pub fn new(points: Vec<P>, builder: Builder) -> Self {
        let mut rng = SmallRng::seed_from_u64(builder.seed);
        println!("seed: {}", builder.seed);

        // Initialize Random Graph
        println!("rand init phase");
        let start_time = Instant::now();
        let mut ann = Vamana::<P>::random_graph_init(points, builder, &mut rng);
        println!(
            "\nrand init time: {:?}",
            Instant::now().duration_since(start_time)
        );

        // Prune Edges
        Vamana::<P>::indexing(&mut ann, &mut rng);

        println!(
            "\ntotal indexing time: {:?}",
            Instant::now().duration_since(start_time)
        );

        ann
    }

    fn random_graph_init(points: Vec<P>, builder: Builder, _rng: &mut SmallRng) -> Self {
        if points.is_empty() {
            return Self {
                nodes: Vec::new(),
                centroid: usize::MAX,
                builder,
            };
        }

        assert!(points.len() < u32::MAX as usize);
        let points_len = points.len();

        /* Find Centroid */
        let mut sum = points[0].clone();
        for p in &points[1..] {
            sum = sum.add(p);
        }

        let average_point = sum.div(&points_len);
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

        // edge (in, out)
        let edges: Vec<(RwLock<Vec<usize>>, RwLock<Vec<usize>>)> = (0..points_len)
            .map(|_| {
                (
                    RwLock::new(Vec::with_capacity(builder.l)),
                    RwLock::new(Vec::with_capacity(builder.l)),
                )
            })
            .collect();

        (0..points_len).into_par_iter().for_each(|node_i| {
            let mut rng = SmallRng::seed_from_u64(builder.seed + node_i as u64);

            let mut new_ids = Vec::with_capacity(builder.l);
            while new_ids.len() < builder.r {
                let candidate_i = rng.gen_range(0..points_len);
                if node_i == candidate_i
                    || new_ids.contains(&candidate_i)
                    || edges[candidate_i].0.read().len() >= builder.r + builder.r / 2
                {
                    continue;
                } else {
                    edges[candidate_i].0.write().push(node_i);
                    new_ids.push(candidate_i);
                }
            }

            let mut n_out = edges[node_i].1.write();
            *n_out = new_ids;
        });

        println!("make nodes");

        let nodes: Vec<Node<P>> = edges
            .into_iter()
            .zip(points)
            .map(|((_n_in, n_out), p)| Node { n_out, p })
            .collect();

        Self {
            nodes,
            centroid,
            builder,
        }
    }

    fn indexing(ann: &mut Vamana<P>, rng: &mut SmallRng) {
        let node_len = ann.nodes.len();
        let mut shuffled: Vec<usize> = (0..node_len).collect();
        shuffled.shuffle(rng);

        // for 1 ≤ i ≤ n do
        shuffled.into_par_iter().enumerate().for_each(|(count, i)| {
            if count % 10000 == 0 {
                println!("id : {}\t/{}", count, ann.nodes.len());
            }

            // let [L; V] ← GreedySearch(s, xσ(i), 1, L)
            let (_, visited) = ann.greedy_search(&ann.nodes[i].p, 1, ann.builder.l);

            // V ← (V ∪ Nout(p)) \ {p}
            let prev_n_out = ann.nodes[i].n_out.read().clone();
            let mut candidates = visited;
            for out_i in &prev_n_out {
                if !is_contained_in(out_i, &candidates) {
                    // let dist = self.node_distance(xp, out_i);
                    let dist = ann.nodes[i].p.distance(&ann.nodes[*out_i].p);
                    insert_dist((dist, *out_i), &mut candidates)
                }
            }

            // run RobustPrune(σ(i), V, α, R) to update out-neighbors of σ(i)
            let mut new_n_out = ann.prune(&mut candidates);
            let new_added_ids = diff_ids(&ann.nodes[i].n_out.read(), &prev_n_out);
            for out_i in new_added_ids {
                insert_id(out_i, &mut new_n_out);
            }

            {
                let mut current_n_out = ann.nodes[i].n_out.write();
                current_n_out.clone_from(&new_n_out);
            } // unlock the write lock

            // for all points j in Nout(σ(i)) do
            for j in new_n_out {
                if ann.nodes[j].n_out.read().contains(&i) {
                    continue;
                } else {
                    // Todo : refactor, self.make_edge　or union. above ann.nodes[j].n_out.contains(&i) not necessary if use union
                    insert_id(i, &mut ann.nodes[j].n_out.write());
                    // insert_id(j, &mut ann.nodes[i].n_in);
                }
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

    fn greedy_search(
        &self,
        query_point: &P,
        k: usize,
        l: usize,
    ) -> (Vec<(f32, usize)>, Vec<(f32, usize)>) {
        // k-anns, visited
        assert!(l >= k);
        let s = self.centroid;
        let mut visited: Vec<(f32, usize)> = Vec::with_capacity(self.builder.l * 2);
        let mut touched = FxHashSet::default();
        touched.reserve(self.builder.l * 100);

        let mut list: Vec<(f32, usize, bool)> = Vec::with_capacity(self.builder.l);
        list.push((query_point.distance(&self.nodes[s].p), s, true));
        let mut working = Some(list[0]);
        visited.push((list[0].0, list[0].1));
        touched.insert(list[0].1);

        while let Some((_, nearest_i, _)) = working {
            let nearest_n_out = self.nodes[nearest_i].n_out.read().clone();
            let mut nouts: Vec<(f32, usize, bool)> = Vec::with_capacity(nearest_n_out.len());
            for out_i in nearest_n_out {
                if !touched.contains(&out_i) {
                    touched.insert(out_i);
                    nouts.push((query_point.distance(&self.nodes[out_i].p), out_i, false))
                }
            }

            sort_list_by_dist(&mut nouts);

            let mut new_list = Vec::with_capacity(self.builder.l);
            let mut new_list_idx = 0;

            let mut l_idx = 0; // Index for list
            let mut n_idx = 0; // Index for dists

            working = None;

            while new_list_idx < self.builder.l {
                let mut new_min = if l_idx >= list.len() && n_idx >= nouts.len() {
                    break;
                } else if l_idx >= list.len() {
                    let new_min = nouts[n_idx];
                    n_idx += 1;
                    new_min
                } else if n_idx >= nouts.len() {
                    let new_min = list[l_idx];
                    l_idx += 1;
                    new_min
                } else {
                    let l_min = list[l_idx];
                    let n_min = nouts[n_idx];

                    if l_min.0 <= n_min.0 {
                        l_idx += 1;
                        l_min
                    } else {
                        n_idx += 1;
                        n_min
                    }
                };

                let is_not_visited = !new_min.2;

                if working.is_none() && is_not_visited {
                    new_min.2 = true; // Mark as visited
                    working = Some(new_min);
                    visited.push((new_min.0, new_min.1));
                }

                new_list.push(new_min);
                new_list_idx += 1;
            }

            list = new_list;
        }

        let mut k_anns = list
            .into_iter()
            .map(|(dist, id, _)| (dist, id))
            .collect::<Vec<(f32, usize)>>();
        k_anns.truncate(k);

        sort_list_by_dist_v1(&mut visited);

        (k_anns, visited)
    }
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

fn sort_list_by_dist(list: &mut Vec<(f32, usize, bool)>) {
    list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less));
}

fn sort_list_by_dist_v1(list: &mut Vec<(f32, usize)>) {
    list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Less));
}

fn is_contained_in(i: &usize, vec: &Vec<(f32, usize)>) -> bool {
    !vec.iter()
        .filter(|(_, id)| *id == *i)
        .collect::<Vec<&(f32, usize)>>()
        .is_empty()
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
    // fn to_f32_vec(&self) -> Vec<f32>;
    // fn from_f32_vec(a: Vec<f32>) -> Self;
    fn add(&self, other: &Self) -> Self;
    fn div(&self, divisor: &usize) -> Self;
}

#[cfg(test)]
mod tests {

    use super::{Point as VPoint, *};

    #[derive(Clone, Debug)]
    struct Point(Vec<u32>);
    impl Point {
        fn to_f32_vec(&self) -> Vec<f32> {
            self.0.iter().map(|v| *v as f32).collect()
        }
        fn from_f32_vec(a: Vec<f32>) -> Self {
            Point(a.into_iter().map(|v| v as u32).collect())
        }
    }
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

        fn add(&self, other: &Self) -> Self {
            Point::from_f32_vec(self.to_f32_vec().into_iter().zip(other.to_f32_vec().into_iter()).map(|(x, y)| x + y).collect())
        }
        fn div(&self, divisor: &usize) -> Self {
            Point::from_f32_vec(self.to_f32_vec().into_iter().map(|v| v / *divisor as f32).collect())
        }
    }

    #[test]
    fn fresh_disk_ann_new_empty() {
        let builder = Builder::default();
        let mut rng = SmallRng::seed_from_u64(builder.seed);

        let ann: Vamana<Point> = Vamana::random_graph_init(Vec::new(), builder, &mut rng);
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
        let ann: Vamana<Point> = Vamana::random_graph_init(points, builder, &mut rng);
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

        let ann: Vamana<Point> = Vamana::new(points, builder);
        let xq = Point(vec![0; Point::dim() as usize]);
        let k = 20;
        let (k_anns, _visited) = ann.greedy_search(&xq, k, l);

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

        let ann: Vamana<Point> = Vamana::random_graph_init(points, builder, &mut rng);
        let xq = Point(vec![0; Point::dim() as usize]);
        let k = 10;
        let (k_anns, _visited) = ann.greedy_search(&xq, k, l);

        println!("k_anns: {:?}", k_anns);

        for i in 0..10 {
            assert_eq!(k_anns[i].1, i);
        }
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

    #[test]
    fn test_is_contained_in() {
        let a = vec![(0.2, 2), (0.1, 1), (0.3, 3), (0.0, 0)];
        assert!(is_contained_in(&0, &a));
        assert!(!is_contained_in(&10, &a));
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
}
