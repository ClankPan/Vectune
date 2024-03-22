# Benchmarking

## on MacOS

[ref](https://github.com/flamegraph-rs/flamegraph?tab=readme-ov-file#dtrace-on-macos)

### Install
```sh
cargo install flamegraph
```

### Bench

```sh
RUSTFLAGS="-C debuginfo=2" cargo flamegraph --root --open --example build_freshvamana
```

Note:
- `--root` is  required to run Dtrace on MacOS.
- `--open` shows the output .svg file with default program.
- `RUSTFLAGS="-C debuginfo=2"` specify `[profile.release] debug = true` without writeing in Cargo.tmol.
