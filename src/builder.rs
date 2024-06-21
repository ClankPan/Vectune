use bit_vec::BitVec;
use rustc_hash::FxHashSet;
use std::collections::HashSet;
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::{rngs::SmallRng, Rng};

use parking_lot::RwLock;
use rayon::prelude::*;

use itertools::Itertools;

#[cfg(feature = "progress-bar")]
use indicatif::ProgressBar;
#[cfg(feature = "progress-bar")]
use std::sync::atomic::{self, AtomicUsize};

use crate::utils::*;
use crate::PointInterface;

/// Builder is a structure and implementation for creating a Vamana graph.
///
// - `a` is the threshold for RobustPrune; increasing it results in more long-distance edges and fewer nearby edges.
// - `r` represents the number of edges; increasing it adds complexity to the graph but reduces the number of isolated nodes.
// - `l` is the size of the retention list for greedy-search; increasing it allows for the construction of more accurate graphs, but the computational cost grows exponentially.
// - `seed` is used for initializing random graphs; it allows for the fixation of the random graph, which can be useful for debugging.
///
#[derive(Clone)]
pub struct Builder {
    a: f32,
    r: usize,
    l: usize,
    seed: u64,

    #[cfg(feature = "progress-bar")]
    progress: Option<ProgressBar>,
}

pub const DEFAULT_R: usize = 70;

impl Default for Builder {
    fn default() -> Self {
        Self {
            a: 2.0,
            r: DEFAULT_R,
            l: 125,
            seed: rand::random(),
            #[cfg(feature = "progress-bar")]
            progress: None,
        }
    }
}

impl Builder {
    pub fn set_a(mut self, a: f32) -> Self {
        self.a = a;
        self
    }
    pub fn set_r(mut self, r: usize) -> Self {
        self.r = r;
        self
    }
    pub fn set_l(mut self, l: usize) -> Self {
        self.l = l;
        self
    }
    pub fn set_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn get_a(&self) -> f32 {
        self.a
    }

    pub fn get_r(&self) -> usize {
        self.r
    }
    pub fn get_l(&self) -> usize {
        self.l
    }
    pub fn get_seed(&self) -> u64 {
        self.seed
    }

    ///
    /// Creates a directed Vamana graph from a Point type that implements the PointInterface.
    ///
    /// Takes a `Vec<P>` as an argument and returns a `Vec<(P, Vec<usize>)>` with edges added.
    ///
    pub fn build<P: PointInterface>(
        self,
        points: Vec<P>,
    ) -> (Vec<(P, Vec<u32>)>, u32, Vec<Vec<u32>>) {
        let ann = Vamana::new(points, self);

        let nodes = ann
            .nodes
            .into_iter()
            .map(|node| {
                (
                    node.p,
                    node.n_out
                        .into_inner()
                        .into_iter()
                        .map(|(_, i)| i)
                        .sorted() // Note: insert_id() requires a sorted array
                        .collect(),
                )
            })
            .collect();
        let s = ann.centroid;

        (nodes, s, ann.backlinks)
    }

    #[cfg(feature = "progress-bar")]
    pub fn progress(mut self, bar: ProgressBar) -> Self {
        self.progress = Some(bar);
        self
    }
}

pub struct Node<P> {
    n_out: RwLock<Vec<(f32, u32)>>,
    p: P,
    nn: RwLock<u32>,
}

pub struct Vamana<P> {
    pub nodes: Vec<Node<P>>,
    pub centroid: u32,
    pub builder: Builder,
    pub backlinks: Vec<Vec<u32>>,
}

impl<P> Vamana<P>
where
    P: PointInterface,
{
    pub fn new(points: Vec<P>, builder: Builder) -> Self {
        let mut rng = SmallRng::seed_from_u64(builder.seed);
        // println!("seed: {}", builder.seed);

        let start_time = Instant::now();
        let mut ann = Vamana::<P>::random_graph_init(points, builder, &mut rng);

        // Prune Edges
        Vamana::<P>::indexing(&mut ann, &mut rng);

        println!(
            "\ntotal indexing time: {:?}",
            Instant::now().duration_since(start_time)
        );

        ann
    }

    pub fn random_graph_init(points: Vec<P>, builder: Builder, _rng: &mut SmallRng) -> Self {
        if points.is_empty() {
            return Self {
                nodes: Vec::new(),
                centroid: u32::MAX,
                builder,
                backlinks: Vec::new(),
            };
        }

        assert!(points.len() < u32::MAX as usize);
        let points_len = points.len();

        /* Find Centroid */
        println!("finding centroid");
        // let mut sum = points[0].clone();
        // for p in &points[1..] {
        //     sum = sum.add(p);
        // }
        // let sum = points.par_iter().reduce_with(|acc, x| &acc.add(x)).unwrap();
        let sum = points
            .par_iter()
            .fold(|| P::zero(), |acc, x| acc.add(x))
            .reduce_with(|sum1, sum2| sum1.add(&sum2))
            .unwrap();

        let average_point = sum.div(&points_len);
        // let mut min_dist = f32::MAX;
        // let mut centroid = u32::MAX;
        // for (i, p) in points.iter().enumerate() {
        //     let dist = p.distance(&average_point);
        //     if dist < min_dist {
        //         min_dist = dist;
        //         centroid = i as u32;
        //     }
        // }
        println!("finding medoid");
        let centroid = points
            .par_iter()
            .enumerate()
            .map(|(i, p)| (i, p.distance(&average_point)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Less))
            .unwrap()
            .0 as u32;

        /* Get random connected graph */

        // edge (in, out)
        println!("init empty edges");
        let edges: Vec<(RwLock<u32>, RwLock<Vec<(f32, u32)>>, RwLock<u32>)> = (0..points_len)
            .into_par_iter()
            .map(|_| {
                (
                    RwLock::new(0),                             // n_in,
                    RwLock::new(Vec::with_capacity(builder.r)), // n_out,
                    RwLock::new(0),                             // nn (nearest node)
                )
            })
            .collect();

        println!("connecting nodes randomly");
        (0..points_len).into_par_iter().for_each(|node_i| {
            let mut rng = SmallRng::seed_from_u64(builder.seed + node_i as u64);

            let mut new_ids = Vec::with_capacity(builder.r);
            while new_ids.len() < builder.r {
                let candidate_i = rng.gen_range(0..points_len as u32);
                if node_i as u32 == candidate_i
                    || new_ids.contains(&candidate_i)
                    || *edges[candidate_i as usize].0.read() >= (builder.r + builder.r / 2) as u32
                {
                    continue;
                } else {
                    *edges[candidate_i as usize].0.write() += 1;
                    new_ids.push(candidate_i);
                }
            }

            let new_n_out: Vec<(f32, u32)> = new_ids
                .into_par_iter()
                .map(|edge_i| {
                    let dist = points[edge_i as usize].distance(&points[node_i]);
                    (dist, edge_i)
                })
                .collect();

            *edges[node_i].1.write() = new_n_out;

            // let mut n_out = edges[node_i].1.write();
            // *n_out = new_ids;
        });

        // println!("make nodes");
        println!("zipping point and edges");
        // let nodes: Vec<Node<P>> = points
        //     .into_par_iter()
        //     .zip(edges.into_par_iter())
        //     .map(|(p, (_n_in, n_out, nn))| {
        //         Node {
        //             n_out,
        //             p,
        //             nn,
        //         }
        //     })
        //     .collect();

        let nodes: Vec<Node<P>> = edges
            .into_par_iter()
            .zip(points)
            .map(|((_n_in, n_out, nn), p)| Node { n_out, p, nn })
            .collect();

        // let nodes: Vec<Node<&P>> = edges
        //     .into_par_iter()
        //     .zip(points.par_iter())
        //     .map(|((_n_in, n_out, nn), p)| {
        //         let n_out: Vec<(f32, u32)> = n_out
        //             .read()
        //             .clone()
        //             .into_iter()
        //             .map(|edge_i| {
        //                 let dist = points[edge_i as usize].distance(&p);
        //                 (dist, edge_i)
        //             })
        //             .collect();
        //         Node {
        //             n_out: RwLock::new(n_out),
        //             p,
        //             nn,
        //         }
        //     })
        //     .collect();

        Self {
            nodes,
            centroid,
            builder,
            backlinks: Vec::new(),
        }
    }

    fn get_missings(ann: &Vamana<P>) -> Vec<u32> {
        ann.nodes
            .par_chunks(1000)
            .map(|chunk| {
                let mut bit_vec = BitVec::from_elem(ann.nodes.len(), false);
                for node in chunk {
                    for (_, edges_i) in &*node.n_out.read() {
                        bit_vec.set(*edges_i as usize, true);
                    }
                }
                bit_vec
            })
            .reduce_with(|mut acc, x| {
                acc.and(&x);
                acc
            })
            .unwrap()
            .into_iter()
            .enumerate()
            .filter_map(|(index, is_true)| if !is_true { Some(index as u32) } else { None })
            .collect()
    }

    fn get_backlinks(ann: &Vamana<P>) -> Vec<Vec<u32>> {
        // Backlinks
        let mut all_edges: Vec<(u32, u32)> = ann
            .nodes
            .iter()
            .enumerate()
            .flat_map(|(i, node)| {
                let i = i as u32;
                node.n_out
                    .read()
                    .clone()
                    .into_iter()
                    .map(move |(_, out_i)| (out_i, i))
            })
            .collect();

        all_edges.par_sort_unstable();

        let node_has_backlinks: Vec<(u32, Vec<u32>)> = all_edges
            .into_iter()
            .group_by(|&(key, _)| key)
            .into_iter()
            .map(|(key, group)| (key, group.map(|(_, edge)| edge).collect()))
            .collect();

            node_has_backlinks
            .into_iter()
            .map(|(_, edges)| edges)
            .collect()
    }

    // fn get_backlinks(ann: &Vamana<P>) -> (Vec<u32>, Vec<Vec<u32>>) {
    //     // Backlinks
    //     let node_has_backlinks: Vec<(u32, Vec<u32>)> = ann
    //         .nodes
    //         .iter()
    //         .enumerate()
    //         .map(|(i, node)| {
    //             let i = i as u32;
    //             node.n_out
    //                 .read()
    //                 .clone()
    //                 .into_iter()
    //                 .map(move |(_, out_i)| (out_i, i))
    //         })
    //         .flatten()
    //         .sorted()
    //         .group_by(|&(key, _)| key)
    //         .into_iter()
    //         .map(|(key, group)| (key, group.map(|(_, edge)| edge).collect()))
    //         .collect();
    //     // let set: HashSet<u32> = node_has_backlinks.iter().map(|(key, _)| *key).collect();
    //     let mut set = BitVec::from_elem(ann.nodes.len(), false);
    //     for key in node_has_backlinks.iter().map(|(key, _)| *key) {
    //         set.set(key as usize, true)
    //     }
    //     // let missings: Vec<u32> = (0..ann.nodes.len() as u32)
    //     //     .filter(|num| !set.contains(num))
    //     //     .collect();
    //     let missings: Vec<u32> = set
    //         .into_iter()
    //         .enumerate()
    //         .filter_map(|(index, is_true)| if !is_true { Some(index as u32) } else { None })
    //         .collect();
    //     // println!("missings, {:?}", missings);

    //     (
    //         missings,
    //         node_has_backlinks
    //             .into_iter()
    //             .map(|(_, edges)| edges)
    //             .collect(),
    //     )
    // }

    pub fn indexing(ann: &mut Vamana<P>, rng: &mut SmallRng) {
        #[cfg(feature = "progress-bar")]
        let progress = &ann.builder.progress;
        #[cfg(feature = "progress-bar")]
        let progress_done = AtomicUsize::new(0);
        #[cfg(feature = "progress-bar")]
        if let Some(bar) = &progress {
            bar.set_length((ann.nodes.len() * 3) as u64);
            bar.set_message("Build index (preparation)");
        }

        let node_len = ann.nodes.len();
        let mut shuffled: Vec<usize> = (0..node_len).collect();
        shuffled.shuffle(rng);

        // for 1 ≤ i ≤ n do
        shuffled
            .into_par_iter()
            .enumerate()
            .for_each(|(_count, i)| {
                // let [L; V] ← GreedySearch(s, xσ(i), 1, L)
                let (_, visited) = ann.greedy_search(&ann.nodes[i].p, 1, ann.builder.l);

                // V ← (V ∪ Nout(p)) \ {p}
                let prev_n_out = ann.nodes[i].n_out.read().clone();
                let mut candidates: Vec<(f32, u32)> = visited;
                for (out_i_dist, out_i) in &prev_n_out {
                    insert_dist((*out_i_dist, *out_i), &mut candidates)
                }

                // run RobustPrune(σ(i), V, α, R) to update out-neighbors of σ(i)
                let mut new_n_out = ann.prune_v2(&mut candidates, vec![]);
                {
                    let mut current_n_out = ann.nodes[i].n_out.write();
                    let new_added_ids = diff_ids(
                        &current_n_out.iter().map(|(_, i)| *i).sorted().collect(),
                        &prev_n_out.iter().map(|(_, i)| *i).sorted().collect(),
                    );
                    for out_i in new_added_ids {
                        let n = current_n_out
                            .iter()
                            .find(|(_, i)| *i == out_i)
                            .unwrap()
                            .clone();
                        insert_dist(n, &mut new_n_out);
                    }
                    sort_list_by_dist_v1(&mut new_n_out);
                    *current_n_out = new_n_out.clone();
                }

                // for all points j in Nout(σ(i)) do
                for (j_dist, j) in new_n_out {
                    if is_contained_in(&(i as u32), &ann.nodes[j as usize].n_out.read()) {
                        continue;
                    } else {
                        // Todo : refactor, self.make_edge　or union. above ann.nodes[j].n_out.contains(&i) not necessary if use union
                        let mut current_n_out = ann.nodes[j as usize].n_out.write();
                        insert_dist((j_dist, i as u32), &mut current_n_out);
                        sort_list_by_dist_v1(&mut current_n_out);
                    }
                }

                #[cfg(feature = "progress-bar")]
                if let Some(bar) = &progress {
                    let value = progress_done.fetch_add(2, atomic::Ordering::Relaxed);
                    if value % 1000 == 0 {
                        bar.set_position(value as u64);
                    }
                }
            });

        // Add node's nearest neigbor
        loop {
            let is_stable = (0..node_len)
                .into_par_iter()
                .map(|node_i| {
                    let nn = ann.nodes[node_i].n_out.read()[0];
                    let is_same = ann.nodes[node_i].nn.read().clone() == nn.1;
                    *ann.nodes[node_i].nn.write() = nn.1;

                    // insert_dist((nn.0, node_i as u32), &mut ann.nodes[nn.1 as usize].n_out.write());

                    let mut current_n_out = ann.nodes[nn.1 as usize].n_out.write();
                    insert_dist((nn.0, node_i as u32), &mut current_n_out);
                    sort_list_by_dist_v1(&mut current_n_out);

                    // #[cfg(feature = "progress-bar")]
                    // if let Some(bar) = &progress {
                    //     let value = progress_done.fetch_add(1, atomic::Ordering::Relaxed);
                    //     if value % 1000 == 0 {
                    //         bar.set_position(value as u64);
                    //     }
                    // }

                    is_same
                })
                .reduce_with(|acc, x| acc & x)
                .unwrap();

            if is_stable {
                break;
            }
        }

        // Vamana::<P>::no_backlinks_nodes(&ann);

        (0..node_len).into_par_iter().for_each(|node_i| {
            let mut n_out = ann.nodes[node_i].n_out.write();

            // let original_n_out_len = n_out.len();

            // sort_list_by_dist_v1(&mut n_out_dist);
            let mut candidates = n_out.clone();
            *n_out = ann.prune_v2(&mut candidates, vec![]);

            // // WIP
            // if n_out.len() > ann.builder.r {
            //     println!(
            //         "node_i: {}: {}, oroginal_len {}",
            //         node_i,
            //         n_out.len(),
            //         original_n_out_len
            //     );
            // }

            #[cfg(feature = "progress-bar")]
            if let Some(bar) = &progress {
                let value = progress_done.fetch_add(1, atomic::Ordering::Relaxed);
                if value % 1000 == 0 {
                    bar.set_position(value as u64);
                }
            }
        });

        #[cfg(feature = "progress-bar")]
        if let Some(bar) = &progress {
            bar.finish();
        }

        // 辿り着けないノードのedgeを全てリセットする。
        // そのノードへの唯一のルートの中で、missingがあると、そのノードも間接的にmissingになる。
        let mut current_missing = Vamana::<P>::get_missings(&ann);
        let missings = loop {
            current_missing
                .clone()
                .into_par_iter()
                .for_each(|missing_i| {
                    *ann.nodes[missing_i as usize].n_out.write() = vec![];
                });
            println!("missings len, {}", current_missing.len());

            let missings_2 = Vamana::<P>::get_missings(&ann);

            if current_missing.len() == missings_2.len() {
                break missings_2;
            }
            current_missing = missings_2;
        };

        // 新しいedgeを与える。
        missings.clone().into_par_iter().for_each(|missing_i| {
            let (_, mut visited) =
                ann.greedy_search(&ann.nodes[missing_i as usize].p, 1, ann.builder.l);
            let n_out_dists = ann.prune_v2(&mut visited, vec![]);
            *ann.nodes[missing_i as usize].n_out.write() = n_out_dists;
        });

        // Nearest-Nodeのedgeの一番最後のedgeと自分を交換する。
        // そのNNを交換したnodeで書き換える。
        // グラフは、missingsがなくても検索が成り立っているので、NNを入れ替えても問題ない。
        missings.into_par_iter().for_each(|missing_i| {
            let mut n_out = ann.nodes[missing_i as usize].n_out.write();
            let (nn_dist, nn_i) = n_out[0];
            let mut nn_n_out = ann.nodes[nn_i as usize].n_out.write();

            let a = &ann.nodes[missing_i as usize].p;

            if nn_n_out.len() == ann.builder.r {
                let (_, poped_i) = nn_n_out.pop().unwrap();
                let dist_a_poped_i = a.distance(&ann.nodes[poped_i as usize].p);
                n_out[0] = (dist_a_poped_i, poped_i);
            }

            insert_dist((nn_dist, missing_i), &mut nn_n_out);
        });

        /*
        メモ：
         missingにvec![]を入れると、それが参照していたやつもmissingになる。
         それらに新しいedgeを入れると、二段階目のやつからmissingが生まれる。
        */

        let backlinks = Vamana::<P>::get_backlinks(&ann);

        ann.backlinks = backlinks;
    }

    pub fn prune(&self, candidates: &mut Vec<(f32, u32)>) -> Vec<u32> {
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
                let pa_point = &self.nodes[pa as usize].p;
                let pd_point = &self.nodes[pd as usize].p;
                let dist_pa_pd = pa_point.distance(pd_point);

                self.builder.a * dist_pa_pd > dist_xp_pd
            })
        }

        new_n_out.into_iter().sorted().collect()
        // new_n_out
    }

    pub fn prune_v2(
        &self,
        candidates: &mut Vec<(f32, u32)>,
        filter: Vec<(f32, u32)>,
    ) -> Vec<(f32, u32)> {
        let mut new_n_out = filter;
        sort_list_by_dist_v1(&mut new_n_out);

        while let Some((first, rest)) = candidates.split_first() {
            let (pa_dist, pa) = *first; // pa is p asterisk (p*), which is nearest point to p in this loop

            if new_n_out.len() == self.builder.r {
                break;
            }

            insert_dist((pa_dist, pa), &mut new_n_out);

            *candidates = rest.to_vec();

            // if α · d(p*, p') <= d(p, p') then remove p' from v
            candidates.retain(|&(dist_xp_pd, pd)| {
                let pa_point = &self.nodes[pa as usize].p;
                let pd_point = &self.nodes[pd as usize].p;
                let dist_pa_pd = pa_point.distance(pd_point);

                self.builder.a * dist_pa_pd > dist_xp_pd
            })
        }

        new_n_out
    }

    pub fn greedy_search(
        &self,
        query_point: &P,
        k: usize,
        l: usize,
    ) -> (Vec<(f32, u32)>, Vec<(f32, u32)>) {
        // k-anns, visited
        assert!(l >= k);
        let s = self.centroid;
        let mut visited: Vec<(f32, u32)> = Vec::with_capacity(self.builder.l * 2);
        let mut touched = FxHashSet::default();
        touched.reserve(self.builder.l * 100);

        let mut list: Vec<(f32, u32, bool)> = Vec::with_capacity(self.builder.l);
        list.push((query_point.distance(&self.nodes[s as usize].p), s, true));
        let mut working = Some(list[0]);
        visited.push((list[0].0, list[0].1));
        touched.insert(list[0].1);

        while let Some((_, nearest_i, _)) = working {
            let nearest_n_out = self.nodes[nearest_i as usize].n_out.read().clone();
            let mut nouts: Vec<(f32, u32, bool)> = Vec::with_capacity(nearest_n_out.len());
            for (_, out_i) in nearest_n_out {
                if !touched.contains(&out_i) {
                    touched.insert(out_i);
                    nouts.push((
                        query_point.distance(&self.nodes[out_i as usize].p),
                        out_i,
                        false,
                    ))
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
            .collect::<Vec<(f32, u32)>>();
        k_anns.truncate(k);

        sort_list_by_dist_v1(&mut visited);

        (k_anns, visited)
    }
}
