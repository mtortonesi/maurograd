# Benchmarks

Three scripts that all time the same thing — one 2D convolution's forward
and backward pass — on three different implementations, so they can be
compared directly:

| Script | Implementation |
|---|---|
| `bench_maurograd_conv.rb` | Maurograd's own `Ops::Conv2D` (im2col + GEMM via the C extension, OpenBLAS) |
| `bench_pytorch_conv.py` | PyTorch's `torch.nn.Conv2d` |
| `bench_torchrb_conv.rb` | [torch-rb](https://github.com/ankane/torch.rb)'s `Torch::NN::Conv2d` (libtorch bindings) |

Each one builds one conv layer, runs a warmup, then times `iters` iterations
of forward + backward separately, and prints a single-line result:

```
{impl: "maurograd_numo", n: 32, cin: 1, cout: 8, h: 28, w: 28, kh: 3, kw: 3,
 stride: 1, pad: 1, iters: 20, fwd_ms_avg: ..., bwd_ms_avg: ..., total_ms_avg: ...}
```

`fwd_ms_avg` / `bwd_ms_avg` / `total_ms_avg` are milliseconds, averaged
across `iters` (after `warmup` untimed iterations to let things settle).

## Running them

All three scripts read the same set of environment variables, with matching
defaults, so the conv shape is apples-to-apples by default:

| Var | Meaning | Default |
|---|---|---|
| `N` | batch size | 32 |
| `CIN` | input channels | 1 |
| `H`, `W` | input height/width | 28, 28 |
| `COUT` | output channels | 8 |
| `KH`, `KW` | kernel height/width | 3, 3 |
| `STRIDE` | convolution stride | 1 |
| `PAD` | zero-padding | 1 |
| `ITERS` | timed iterations | differs per script - see below |
| `WARMUP` | untimed iterations before timing starts | differs per script - see below |

`ITERS`/`WARMUP` default differently per script (maurograd: 20/5, torch-rb:
50/10, PyTorch: 200/50) since each implementation needs a different number
of iterations to get a stable reading. Set them explicitly to the same
values on all three if you want a strictly matched comparison.

```bash
# Maurograd (always available - no extra setup beyond the main Gemfile)
bundle exec ruby bench/bench_maurograd_conv.rb

# torch-rb (needs the :bench Gemfile group: bundle install, no --without bench)
bundle exec ruby bench/bench_torchrb_conv.rb

# PyTorch (needs a Python environment with torch installed)
python bench/bench_pytorch_conv.py

# Same shape, explicit iteration counts, across all three:
N=64 COUT=16 ITERS=50 WARMUP=10 bundle exec ruby bench/bench_maurograd_conv.rb
N=64 COUT=16 ITERS=50 WARMUP=10 bundle exec ruby bench/bench_torchrb_conv.rb
N=64 COUT=16 ITERS=50 WARMUP=10 python bench/bench_pytorch_conv.py
```

## Reproducible comparisons: `env_openblas.sh`

By default, Maurograd lets OpenBLAS use as many threads as it wants (see
the roadmap's Phase 0 fix removing the old hardcoded single-thread pin) -
which is what you want in real use, but makes run-to-run timings noisier
and not directly comparable to single-threaded baselines. `env_openblas.sh`
pins OpenBLAS/OMP/MKL/vecLib to a single thread for the duration of a
command:

```bash
bench/env_openblas.sh bundle exec ruby bench/bench_maurograd_conv.rb
```

Use it on both sides of a comparison (or neither) - mixing a pinned run
against an unpinned one isn't a fair comparison.
