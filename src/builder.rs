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

impl Default for Builder {
    fn default() -> Self {
        Self {
            a: 2.0,
            r: 70,
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
    pub fn build<P: PointInterface>(self, points: Vec<P>) -> (Vec<(P, Vec<u32>)>, u32) {
        let ann = Vamana::new(points, self);

        let nodes = ann
            .nodes
            .into_iter()
            .map(|node| {
                (
                    node.p,
                    node.n_out.into_inner().into_iter().sorted().collect(),
                )
            })
            .collect();
        let s = ann.centroid;

        (nodes, s)
    }

    #[cfg(feature = "progress-bar")]
    pub fn progress(mut self, bar: ProgressBar) -> Self {
        self.progress = Some(bar);
        self
    }
}

pub struct Node<P> {
    n_out: RwLock<Vec<u32>>,
    p: P,
}

pub struct Vamana<P> {
    pub nodes: Vec<Node<P>>,
    pub centroid: u32,
    pub builder: Builder,
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
        let mut centroid = u32::MAX;
        for (i, p) in points.iter().enumerate() {
            let dist = p.distance(&average_point);
            if dist < min_dist {
                min_dist = dist;
                centroid = i as u32;
            }
        }

        /* Get random connected graph */

        // edge (in, out)
        let edges: Vec<(RwLock<Vec<u32>>, RwLock<Vec<u32>>)> = (0..points_len)
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
                let candidate_i = rng.gen_range(0..points_len as u32);
                if node_i as u32 == candidate_i
                    || new_ids.contains(&candidate_i)
                    || edges[candidate_i as usize].0.read().len() >= builder.r + builder.r / 2
                {
                    continue;
                } else {
                    edges[candidate_i as usize].0.write().push(node_i as u32);
                    new_ids.push(candidate_i);
                }
            }

            let mut n_out = edges[node_i].1.write();
            *n_out = new_ids;
        });

        // println!("make nodes");

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

    fn no_backlinks_nodes(ann: &Vamana<P>) {
        // Backlinks
        let node_has_backlinks: Vec<u32> = ann
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let i = i as u32;
                node.n_out
                    .read()
                    .clone()
                    .into_iter()
                    .map(move |out_i| (out_i, i))
            })
            .flatten()
            .sorted()
            .group_by(|&(key, _)| key)
            .into_iter()
            .map(|(key, _group)| {
                key
            })
            .collect();
            let set: HashSet<u32> = node_has_backlinks.into_iter().collect();
            let missings: Vec<u32> = (0..ann.nodes.len() as u32).filter(|num| !set.contains(num)).collect();
            println!("missings, {:?}", missings);
    }

    pub fn indexing(ann: &mut Vamana<P>, rng: &mut SmallRng) {
        #[cfg(feature = "progress-bar")]
        let progress = &ann.builder.progress;
        #[cfg(feature = "progress-bar")]
        let progress_done = AtomicUsize::new(0);
        #[cfg(feature = "progress-bar")]
        if let Some(bar) = &progress {
            bar.set_length((ann.nodes.len() * 2) as u64);
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
                // if count % 10000 == 0 {
                //     println!("id : {}\t/{}", count, ann.nodes.len());
                // }

                // let [L; V] ← GreedySearch(s, xσ(i), 1, L)
                let (_, visited) = ann.greedy_search(&ann.nodes[i].p, 1, ann.builder.l);

                // V ← (V ∪ Nout(p)) \ {p}
                let prev_n_out = ann.nodes[i].n_out.read().clone();
                let mut candidates = visited;
                for out_i in &prev_n_out {
                    if !is_contained_in(out_i, &candidates) {
                        // let dist = self.node_distance(xp, out_i);
                        let dist = ann.nodes[i].p.distance(&ann.nodes[*out_i as usize].p);
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
                    if ann.nodes[j as usize].n_out.read().contains(&(i as u32)) {
                        continue;
                    } else {
                        // Todo : refactor, self.make_edge　or union. above ann.nodes[j].n_out.contains(&i) not necessary if use union
                        insert_id(i as u32, &mut ann.nodes[j as usize].n_out.write());
                        // insert_id(j, &mut ann.nodes[i].n_in);
                    }
                }

                #[cfg(feature = "progress-bar")]
                if let Some(bar) = &progress {
                    let value = progress_done.fetch_add(1, atomic::Ordering::Relaxed);
                    if value % 1000 == 0 {
                        bar.set_position(value as u64);
                    }
                }
            });
            
        // Vamana::<P>::no_backlinks_nodes(&ann);

        (0..node_len).into_par_iter().for_each(|node_i| {
            let node_p = &ann.nodes[node_i].p;
            let mut n_out_dist = ann.nodes[node_i]
                .n_out
                .write()
                .clone()
                .into_iter()
                .map(|out_i| (node_p.distance(&ann.nodes[out_i as usize].p), out_i))
                .collect();

            *ann.nodes[node_i].n_out.write() = ann.prune(&mut n_out_dist);

            #[cfg(feature = "progress-bar")]
            if let Some(bar) = &progress {
                let value = progress_done.fetch_add(1, atomic::Ordering::Relaxed);
                if value % 1000 == 0 {
                    bar.set_position(value as u64);
                }
            }
        });

        // Vamana::<P>::no_backlinks_nodes(&ann);

        #[cfg(feature = "progress-bar")]
        if let Some(bar) = &progress {
            bar.finish();
        }
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
            for out_i in nearest_n_out {
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