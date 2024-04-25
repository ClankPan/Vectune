# Vectune: fast Vamana indexing

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE-MIT)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE-APACHE)


Vectune is a lightweight VectorDB with Incremental Indexing, based on [FreshVamana](https://arxiv.org/pdf/2105.09613.pdf).
This project is implemented with the support of KinicDAO and powers the backend of [KinicVectorDB](https://xcvai-qiaaa-aaaak-afowq-cai.icp0.io/) for vector indexing.

## Getting Start

By specifying progress-bar in features, you can check the progress of indexing.

```toml
[dependencies]
vectune = {version = "0.1.0", features = ["progress-bar"]}
```

To perform calculations of Euclidean distances quickly using SIMD, it is necessary to specify `nightly` in example. If the `rust-analyzer` in VSCode gives an error for `#![feature(portable_simd)]`, please set up your `.vscode/settings.json`.

```json
{
  "rust-analyzer.server.extraEnv": {
      "RUSTUP_TOOLCHAIN": "nightly"
  },
}
```

## Example

### Setup and Run

To test with the SIFT1M dataset, please execute the following command. SIFT1M is a dataset of 1 million data points, each with 128 dimensions.

```bash
curl ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz -o examples/test_data/sift.tar.gz
tar -xzvf examples/test_data/sift.tar.gz -C examples/test_data

cargo +nightly run --release --features progress-bar --example sift1m
```

### How it works

Indexing is performed on the data using a Builder, and searches and insertions are conducted on the graph.

```rust
use vectune::{Builder, GraphInterface, PointInterface};

let points = Vec::new();
for vec in base_vectors {
    points.push(Point(vec.to_vec()));
}

let (nodes, centroid) = Builder::default()
    .progress(ProgressBar::new(1000))
    .build(points);

let mut graph = Graph::new(nodes, centroid);

let k = 50;

let (top_k_results, _visited) = vectune::search(&mut graph, &Point(query.to_vec()), k);
```

### PointInterface Trait

You will need to define the dimensions and data type of the vectors used, as well as the method for calculating distance.

Please implement the following four methods:
- `distance(&self, other: &Self) -> f32`
- `fn dim() -> u32`
- `fn add(&self, other: &Self) -> Self`
- `fn div(&self, divisor: &usize) -> Self`

`distance()` can be optimized using SIMD. Please refer to `./examples/src/bin/sift1m.rs`.

The following example provides a simple implementation.


```rust
use vectune::PointInterface;

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Point(Vec<f32>);
impl Point {
    fn to_f32_vec(&self) -> Vec<f32> {
        self.0.iter().copied().collect()
    }
    fn from_f32_vec(a: Vec<f32>) -> Self {
        Point(a.into_iter().collect())
    }
}
impl PointInterface for Point {
    fn distance(&self, other: &Self) -> f32 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| {
                let c = a - b;
                c * c
            })
            .sum::<f32>()
            .sqrt()
    }
    fn dim() -> u32 {
        384
    }
    fn add(&self, other: &Self) -> Self {
        Point::from_f32_vec(
            self.to_f32_vec()
                .into_iter()
                .zip(other.to_f32_vec().into_iter())
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
}
```


### GraphInterface Trait

To accommodate the entire graph on storage solutions other than SSDs or other memory types, you need to implement the `GraphInterface`.

Please implement the following eleven methods:
- `fn alloc(&mut self, point: P) -> usize`
- `fn free(&mut self, id: &usize)`
- `fn cemetery(&self) -> Vec<usize>`
- `fn clear_cemetery(&mut self)`
- `fn backlink(&self, id: &usize) -> Vec<usize>`
- `fn get(&mut self, id: &usize) -> (P, Vec<usize>)`
- `fn size_l(&self) -> usize`
- `fn size_r(&self) -> usize`
- `fn size_a(&self) -> f32`
- `fn start_id(&self) -> usize`
- `fn overwirte_out_edges(&mut self, id: &usize, edges: Vec<usize>)`

`self.get()` is defined with `&mut self` because it handles caching from SSDs and other storage devices.

In `vectune::search()`, nodes returned by `self.cemetery()` are marked as tombstones and are excluded from the search results. Additionally, they are permanently deleted in `vectune::delete()`.

You need to manage backlinks when adding or deleting nodes. This is utilized in `vectune::delete()`.

The following example provides a simple on-memory implementation.


```rust
use vectune::GraphInterface;
use itertools::Itertools;

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

```

## Indexing

- `a` is the threshold for RobustPrune; increasing it results in more long-distance edges and fewer nearby edges.
- `r` represents the number of edges; increasing it adds complexity to the graph but reduces the number of isolated nodes.
- `l` is the size of the retention list for greedy-search; increasing it allows for the construction of more accurate graphs, but the computational cost grows exponentially.
- `seed` is used for initializing random graphs; it allows for the fixation of the random graph, which can be useful for debugging.

```rust
let (nodes, centroid) = Builder::default()
    .set_a(2.0)
    .set_r(70)
    .set_l(125)
    .set_seed(11677721592066047712)
    .progress(ProgressBar::new(1000))
    .build(points);
```

## Searching

`k` represents the number of top-k results. It is necessary that `k <= l`.

```rust
vectune::search(&mut graph, &point, k);
```

## Inserting

```rust
vectune::insert(&mut graph, point);
```

## Deleting

Completely remove the nodes returned by `graph.cemetery()` from the graph.

```rust
vectune::delete(&mut graph);
```

## Ordering

Reordering the arrangement to efficiently reference nodes from storage such as SSDs.
This algorithm is proposed in Section 4 of this [paper](https://arxiv.org/pdf/2211.12850v2.pdf). 

```rust
vectune::gorder(
    edges,      // Vec<Vec<u32>>
    backlinks,  // Vec<Vec<u32>>
    10,         // Number of nodes in one section
    &mut rng,
);
```