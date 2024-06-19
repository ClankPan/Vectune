use super::{GraphInterface as VGraph, PointInterface as VPoint, *};
use bit_vec::BitVec;
use itertools::Itertools;
use rand::rngs::SmallRng;
use rand::SeedableRng;

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
        Point::from_f32_vec(
            self.to_f32_vec()
                .into_iter()
                .zip(other.to_f32_vec())
                .map(|(x, y)| x + y)
                .collect(),
        )
    }
    fn div(&self, divisor: &usize) -> Self {
        Point::from_f32_vec(
            self.to_f32_vec()
                .into_iter()
                .map(|v| v / *divisor as f32)
                .collect(),
        )
    }

    fn zero() -> Self {
        Point::from_f32_vec(vec![0.0; Point::dim() as usize])
    }
}

struct Graph<P>
where
    P: VPoint,
{
    nodes: Vec<(P, Vec<u32>)>,
    backlinks: Vec<Vec<u32>>,
    cemetery: Vec<u32>,
    centroid: u32,
}

impl<P> VGraph<P> for Graph<P>
where
    P: VPoint,
{
    fn alloc(&mut self, point: P) -> u32 {
        self.nodes.push((point, vec![]));
        self.backlinks.push(vec![]);
        (self.nodes.len() - 1) as u32
    }

    fn free(&mut self, _id: &u32) {
        // todo!()
    }

    fn cemetery(&self) -> Vec<u32> {
        self.cemetery.clone()
    }

    fn clear_cemetery(&mut self) {
        self.cemetery = Vec::new();
    }

    fn backlink(&self, id: &u32) -> Vec<u32> {
        self.backlinks[*id as usize].clone()
    }

    fn get(&mut self, id: &u32) -> (P, Vec<u32>) {
        let node = &self.nodes[*id as usize];
        node.clone()
    }

    fn size_l(&self) -> usize {
        125
    }

    fn size_r(&self) -> usize {
        70
    }

    fn size_a(&self) -> f32 {
        2.0
    }

    fn start_id(&self) -> u32 {
        self.centroid
    }

    fn overwirte_out_edges(&mut self, id: &u32, edges: Vec<u32>) {
        for out_i in &self.nodes[*id as usize].1 {
            let backlinks = &mut self.backlink(out_i);
            backlinks.retain(|out_i| out_i != id)
        }

        for out_i in &edges {
            let backlinks = &mut self.backlink(out_i);
            backlinks.push(*id);
            backlinks.sort();
            backlinks.dedup();
        }

        self.nodes[*id as usize].1 = edges;
    }
}

fn gen_backlinks(nodes: &Vec<(Point, Vec<u32>)>) -> Vec<Vec<u32>> {
    let backlinks: Vec<Vec<u32>> = nodes
        .iter()
        .enumerate()
        .flat_map(|(node_i, node)| {
            node.1
                .iter()
                .map(|out_i| (*out_i, node_i))
                .collect::<Vec<_>>()
        })
        .sorted_by_key(|&(k, _)| k)
        .group_by(|&(k, _)| k)
        .into_iter()
        .map(|(_key, group)| {
            group
                .into_iter()
                .map(|(_, i)| i as u32)
                .sorted()
                .collect::<Vec<u32>>()
        })
        .collect();
    backlinks
}

#[test]
fn test_parallel_gorder() {
    let builder = Builder::default();
    // let builder = Builder::default().set_seed(10910418820652569485);
    let mut rng = SmallRng::seed_from_u64(builder.get_seed());

    println!("seed: {}", builder.get_seed());

    let mut i = 0;

    let points: Vec<Point> = (0..105)
        .map(|_| {
            let a = i;
            i += 1;
            Point(vec![a; Point::dim() as usize])
        })
        .collect();

    let (nodes, centroid, _) = builder.build(points);

    // for (node_i, node) in nodes.iter().enumerate() {
    //     println!("id: {}, {:?}", node_i, node.1);
    // }

    let backlinks: Vec<Vec<u32>> = gen_backlinks(&nodes);

    let mut original_graph = Graph {
        nodes: nodes.clone(),
        backlinks: backlinks.clone(),
        cemetery: Vec::new(),
        centroid,
    };

    // let ordered_nodes = super::gorder(
    //     nodes.iter().map(|(_, outs)| outs.clone()).collect(),
    //     backlinks,
    //     BitVec::from_elem(nodes.len(), false),
    //     10,
    // );
    let get_edges = |id: &u32| -> Vec<u32> { nodes[*id as usize].1.clone() };
    let get_backlinks = |id: &u32| -> Vec<u32> { backlinks[*id as usize].clone() };
    let ordered_nodes: Vec<u32> = super::gorder(
        get_edges,
        get_backlinks,
        BitVec::from_elem(nodes.len(), true),
        10,
        &mut rng,
    )
    .into_iter()
    .flatten()
    .collect();

    println!("ordered_nodes: {:?}\n", ordered_nodes.iter().sorted());

    assert_eq!(nodes.len(), ordered_nodes.len());

    // Create a conversion table from ordered_nodes to original_index->ordered_index.
    let ordered_table: Vec<u32> = ordered_nodes
        .iter()
        .enumerate()
        .map(|(ordered_index, original_index)| (original_index, ordered_index as u32))
        .sorted()
        .map(|(_, ordered_index)| ordered_index)
        .collect();
    // Replace all, including P
    let nodes: Vec<(Point, Vec<u32>)> = ordered_nodes
        .into_iter()
        .map(|original_index| {
            let (p, outs) = nodes[original_index as usize].clone();
            let outs: Vec<u32> = outs
                .into_iter()
                .map(|original_index| ordered_table[original_index as usize])
                .collect();
            (p, outs)
        })
        .collect();

    let centroid = ordered_table[centroid as usize];

    let backlinks: Vec<Vec<u32>> = gen_backlinks(&nodes);

    for (node_i, node) in nodes.iter().enumerate() {
        println!("id: {}, {:?}", node_i, node.1);
    }

    let mut graph = Graph {
        nodes,
        backlinks,
        cemetery: Vec::new(),
        centroid,
    };

    let xq = Point(vec![0; Point::dim() as usize]);
    let k = 10;
    // let (k_anns, _visited) = ann.greedy_search(&xq, k, l);
    let (k_anns, _visited) = super::search(&mut graph, &xq, k);

    let expected_k_anns = super::search(&mut original_graph, &xq, k).0;

    println!("k_anns:\t\t{:?}\noriginal:\t{:?}", k_anns, expected_k_anns);

    for i in 0..10 {
        assert_eq!(k_anns[i].0, expected_k_anns[i].0);
    }
}

#[test]
fn fresh_disk_ann_new_empty() {
    let builder = Builder::default();
    let mut rng = SmallRng::seed_from_u64(builder.get_seed());

    let ann: Vamana<Point> = Vamana::random_graph_init(Vec::new(), builder, &mut rng);
    assert_eq!(ann.nodes.len(), 0);
}

#[test]
fn fresh_disk_ann_new_centroid() {
    let builder = Builder::default();
    let mut rng = SmallRng::seed_from_u64(builder.get_seed());

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
    let builder = Builder::default();
    // builder.set_seed(11677721592066047712);
    let l = builder.get_l();

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
        assert_eq!(k_anns[i].1, i as u32);
    }
}

#[test]
fn search_api() {
    let builder = Builder::default();
    println!("seed: {}", builder.get_seed());

    let mut i = 0;

    let points: Vec<Point> = (0..500)
        .map(|_| {
            let a = i;
            i += 1;
            Point(vec![a; Point::dim() as usize])
        })
        .collect();

    let (nodes, centroid, _) = builder.build(points);

    let mut graph = Graph {
        nodes,
        backlinks: Vec::new(),
        cemetery: Vec::new(),
        centroid,
    };

    // let ann: Vamana<Point> = Vamana::random_graph_init(points, builder, &mut rng);
    let xq = Point(vec![0; Point::dim() as usize]);
    let k = 10;
    // let (k_anns, _visited) = ann.greedy_search(&xq, k, l);
    let (k_anns, _visited) = super::search(&mut graph, &xq, k);

    println!("k_anns: {:?}", k_anns);

    for i in 0..10 {
        assert_eq!(k_anns[i].1, i as u32);
    }
}

#[test]
fn test_greedy_search_with_cemetery() {
    let builder = Builder::default();
    println!("seed: {}", builder.get_seed());

    let mut i = 0;

    let points: Vec<Point> = (0..500)
        .map(|_| {
            let a = i;
            i += 1;
            Point(vec![a; Point::dim() as usize])
        })
        .collect();

    let (nodes, centroid, _) = builder.build(points);

    let mut graph = Graph {
        nodes,
        backlinks: Vec::new(),
        cemetery: Vec::new(),
        centroid,
    };

    // let ann: Vamana<Point> = Vamana::random_graph_init(points, builder, &mut rng);
    let xq = Point(vec![0; Point::dim() as usize]);
    let k = 10;
    // let (k_anns, _visited) = ann.greedy_search(&xq, k, l);
    let (k_anns, _visited) = super::search(&mut graph, &xq, k);

    println!("k_anns: {:?}", k_anns);

    for i in 0..10 {
        assert_eq!(k_anns[i].1, i as u32);
    }

    // mark as grave
    graph.cemetery.push(k_anns[3].1);
    graph.cemetery.push(k_anns[5].1);
    graph.cemetery.push(k_anns[9].1);

    let expected = vec![k_anns[3].1, k_anns[5].1, k_anns[9].1];

    let (k_anns_intered, _visited) = super::search(&mut graph, &xq, k);

    assert_ne!(k_anns_intered, k_anns);

    let k_anns_ids: Vec<u32> = k_anns.into_iter().map(|(_, id)| id).collect();
    let k_anns_intered_ids: Vec<u32> = k_anns_intered.into_iter().map(|(_, id)| id).collect();

    let diff = diff_ids(&k_anns_ids, &k_anns_intered_ids);
    assert_eq!(diff, expected);
}

#[test]
fn test_greedy_search_with_removing_graves() {
    let builder = Builder::default();
    println!("seed: {}", builder.get_seed());

    let mut i = 0;

    let points: Vec<Point> = (0..100)
        .map(|_| {
            let a = i;
            i += 1;
            Point(vec![a; Point::dim() as usize])
        })
        .collect();

    let (nodes, centroid, _) = builder.build(points);

    for (node_i, node) in nodes.iter().enumerate() {
        println!("id: {}, {:?}", node_i, node.1);
    }

    let backlinks: Vec<Vec<u32>> = nodes
        .iter()
        .enumerate()
        .flat_map(|(node_i, node)| {
            node.1
                .iter()
                .map(|out_i| (*out_i, node_i))
                .collect::<Vec<_>>()
        })
        .sorted_by_key(|&(k, _)| k)
        .group_by(|&(k, _)| k)
        .into_iter()
        .map(|(_key, group)| {
            group
                .into_iter()
                .map(|(_, i)| i as u32)
                .sorted()
                .collect::<Vec<u32>>()
        })
        .collect();

    let mut graph = Graph {
        nodes,
        backlinks,
        cemetery: Vec::new(),
        centroid,
    };

    // let ann: Vamana<Point> = Vamana::random_graph_init(points, builder, &mut rng);
    let xq = Point(vec![0; Point::dim() as usize]);
    let k = 10;
    // let (k_anns, _visited) = ann.greedy_search(&xq, k, l);
    let (k_anns, _visited) = super::search(&mut graph, &xq, k);

    println!("k_anns: {:?}", k_anns);

    for i in 0..10 {
        assert_eq!(k_anns[i].1, i as u32);
    }

    // mark as grave
    graph.cemetery.push(k_anns[3].1);
    graph.cemetery.push(k_anns[5].1);
    graph.cemetery.push(k_anns[9].1);

    let expected = vec![k_anns[3].1, k_anns[5].1, k_anns[9].1];

    super::delete(&mut graph);

    let (k_anns_intered, _visited) = super::search(&mut graph, &xq, k);

    for (node_i, node) in graph.nodes.iter().enumerate() {
        println!("id: {}, {:?}", node_i, node.1);
    }

    assert_ne!(k_anns_intered, k_anns);

    let k_anns_ids: Vec<u32> = k_anns.into_iter().map(|(_, id)| id).collect();
    let k_anns_intered_ids: Vec<u32> = k_anns_intered.into_iter().map(|(_, id)| id).collect();

    let diff = diff_ids(&k_anns_ids, &k_anns_intered_ids);
    assert_eq!(diff, expected);
}

#[test]
fn test_insert_new_point() {
    let builder = Builder::default();
    println!("seed: {}", builder.get_seed());

    let mut i = 0;

    let points: Vec<Point> = (0..100)
        .map(|_| {
            let a = i;
            i += 1;
            Point(vec![a; Point::dim() as usize])
        })
        .collect();

    let (nodes, centroid, _) = builder.build(points);

    for (node_i, node) in nodes.iter().enumerate() {
        println!("id: {}, {:?}", node_i, node.1);
    }

    let backlinks: Vec<Vec<u32>> = nodes
        .iter()
        .enumerate()
        .flat_map(|(node_i, node)| {
            node.1
                .iter()
                .map(|out_i| (*out_i, node_i))
                .collect::<Vec<_>>()
        })
        .sorted_by_key(|&(k, _)| k)
        .group_by(|&(k, _)| k)
        .into_iter()
        .map(|(_key, group)| {
            group
                .into_iter()
                .map(|(_, i)| i as u32)
                .sorted()
                .collect::<Vec<u32>>()
        })
        .collect();

    let mut graph = Graph {
        nodes,
        backlinks,
        cemetery: Vec::new(),
        centroid,
    };

    // let ann: Vamana<Point> = Vamana::random_graph_init(points, builder, &mut rng);
    let xq = Point(vec![0; Point::dim() as usize]);
    let k = 10;
    // let (k_anns, _visited) = ann.greedy_search(&xq, k, l);
    let (k_anns, _visited) = super::search(&mut graph, &xq, k);

    println!("k_anns: {:?}", k_anns);

    for i in 0..10 {
        assert_eq!(k_anns[i].1, i as u32);
    }

    // mark as grave
    graph.cemetery.push(k_anns[3].1);
    graph.cemetery.push(k_anns[5].1);
    graph.cemetery.push(k_anns[9].1);

    let expected = vec![k_anns[3].1, k_anns[5].1, k_anns[9].1];
    let expected_p = vec![
        graph.nodes[3].0.clone(),
        graph.nodes[5].0.clone(),
        graph.nodes[9].0.clone(),
    ];

    super::delete(&mut graph);

    let (k_anns_intered, _visited) = super::search(&mut graph, &xq, k);

    for (node_i, node) in graph.nodes.iter().enumerate() {
        println!("id: {}, {:?}", node_i, node.1);
    }

    assert_ne!(k_anns_intered, k_anns);

    let mut k_anns_ids: Vec<u32> = k_anns.into_iter().map(|(_, id)| id).collect();
    let k_anns_intered_ids: Vec<u32> = k_anns_intered.into_iter().map(|(_, id)| id).collect();

    let diff = diff_ids(&k_anns_ids, &k_anns_intered_ids);
    assert_eq!(diff, expected);

    let mut new_ids = vec![];
    for new_point in expected_p {
        let new_id = super::insert(&mut graph, new_point);
        new_ids.push(new_id)
    }

    let (k_anns_inserted, _visited) = super::search(&mut graph, &xq, k);
    let k_anns_inserted_ids: Vec<u32> = k_anns_inserted.into_iter().map(|(_, id)| id).collect();
    k_anns_ids[3] = new_ids[0];
    k_anns_ids[5] = new_ids[1];
    k_anns_ids[9] = new_ids[2];
    assert_eq!(k_anns_ids, k_anns_inserted_ids);
}

#[test]
fn greedy_search() {
    let builder = Builder::default();
    println!("seed: {}", builder.get_seed());
    let seed = builder.get_seed();
    // let seed: u64 = 17674802184506369839;
    let mut rng = SmallRng::seed_from_u64(seed);
    let l = builder.get_l();

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
        assert_eq!(k_anns[i].1, i as u32);
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
