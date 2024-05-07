/*
    Vectune is a lightweight VectorDB with Incremental Indexing, based on [FreshVamana](https://arxiv.org/pdf/2105.09613.pdf).
    Copyright © ClankPan 2024.
*/

// use itertools::Itertools;
// use rand::rngs::SmallRng;
// use rand::seq::SliceRandom;
// use rayon::iter::IntoParallelIterator;
// use rayon::iter::ParallelIterator;
use rustc_hash::FxHashSet;
// use std::collections::BinaryHeap;
// use std::sync::atomic::{AtomicBool, Ordering};

pub mod builder;
pub mod gorder;
pub mod traits;
pub mod utils;

#[cfg(test)]
mod tests;

pub use crate::builder::*;
pub use crate::gorder::gorder;
pub use crate::traits::GraphInterface;
pub use crate::traits::PointInterface;
use crate::utils::*;

/// Performs Greedy-Best-First-Search on a Graph that implements the GraphInterface trait.
///
/// Returns a tuple containing the list of k search results and the list of explored nodes.
///
/// Removes the nodes returned by graph.cemetery() from the results.
///
/// # Examples
///
/// ```ignore
/// let (results, visited) = vectune::search(&mut graph, &Point(query), 50);
/// ```
///
pub fn search<P, G>(graph: &mut G, query_point: &P, k: usize) -> (Vec<(f32, u32)>, Vec<(f32, u32)>)
where
    P: PointInterface,
    G: GraphInterface<P>,
{
    // k-anns, visited
    let builder_l = graph.size_l();
    assert!(builder_l >= k);

    let mut visited: Vec<(f32, u32)> = Vec::with_capacity(builder_l * 2);
    let mut touched = FxHashSet::default();
    touched.reserve(builder_l * 100);

    let mut list: Vec<(f32, u32, bool)> = Vec::with_capacity(builder_l);
    let s = graph.start_id();
    let (s_point, _) = graph.get(&s);
    // list.push((query_point.distance(&s_point), s, true));
    list.push((query_point.distance(&s_point), s, true));
    let mut working = Some(list[0]);
    visited.push((list[0].0, list[0].1));
    touched.insert(list[0].1);

    while let Some((_, nearest_i, _)) = working {
        let (_, nearest_n_out) = graph.get(&nearest_i);
        let mut nouts: Vec<(f32, u32, bool)> = Vec::with_capacity(nearest_n_out.len());
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
        .collect::<Vec<(f32, u32)>>();
    k_anns.truncate(k);

    sort_list_by_dist_v1(&mut visited);

    (k_anns, visited)
}

/// Insert a new node into a Graph that implements the GraphInterface trait.
///
/// Internally, use graph.alloc() to allocate space in storage or memory and reconnect the edges.
pub fn insert<P, G>(graph: &mut G, new_p: P) -> u32
where
    P: PointInterface,
    G: GraphInterface<P>,
{
    let new_id = graph.alloc(new_p.clone());
    let r = graph.size_r();
    let a = graph.size_a();

    // [L, V] ← GreedySearch(𝑠, 𝑝, 1, 𝐿)
    let (_list, mut visited) = search(graph, &new_p, 1);
    // 𝑁out(𝑝) ← RobustPrune(𝑝, V, 𝛼, 𝑅) (Algorithm 3)
    let n_out = prune(|id| graph.get(id), &mut visited, &r, &a);

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
                .collect::<Vec<(f32, u32)>>();
            sort_list_by_dist_v1(&mut j_n_out_with_dist);
            j_n_out = prune(|id| graph.get(id), &mut j_n_out_with_dist, &r, &a);
        }
        graph.overwirte_out_edges(j, j_n_out);
    }

    new_id
}

/// Completely removes the nodes returned by graph.cemetery() from a Graph that implements the GraphInterface trait.
pub fn delete<P, G>(graph: &mut G)
where
    P: PointInterface,
    G: GraphInterface<P>,
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
        let mut c_with_dist: Vec<(f32, u32)> = c
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
        let r = graph.size_r();
        let a = graph.size_a();
        let new_edges = prune(|id| graph.get(id), &mut c_with_dist, &r, &a);
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
    mut get: F,
    candidates: &mut Vec<(f32, u32)>,
    builder_r: &usize,
    builder_a: &f32,
) -> Vec<u32>
where
    P: PointInterface,
    F: FnMut(&u32) -> (P, Vec<u32>),
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
