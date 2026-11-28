# Maurograd

Maurograd is my take on creating a framwork for Artificial Neural Networks
(ANNs) in the Ruby programming language.

It started as an exercise to develop a clone of Andrej Karpathy's
[micrograd](https://github.com/karpathy/micrograd) but it almost immediately
became something more ambitious. I wanted to add support for Convolutional
Neural Networks (CNNs) and to have a reasonably CPU-competent framework. So,
after many experiments (and the corresponding performance evaluations) I ended
up with a relatively clean mostly-Ruby codebase that leverages
[numo-narray-alt](https://github.com/yoshoku/numo-narray-alt) as a tensor
framwork and implements 2D convolution in C using the im2col + GEMM + col2im
approach. For simplicity, GEMM is performed with
[OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) (or Apple's Accelerate
framework on macOS). (In general, I hate the
idea of developing C extensions for Ruby, but they are perfect for this task.)

The original scalar-valued micrograd clone this project started as now lives
on its own, unchanged, in
[maurograd_scalar](https://github.com/mtortonesi/maurograd_scalar) — worth a
look if you want to see backpropagation with nothing but plain Ruby arithmetic
in the way.


## Installation

```sh
gem install maurograd
```

Maurograd compiles a C extension on install, so it needs a C compiler and a
BLAS library:

- **macOS**: the Xcode command line tools (`xcode-select --install`). The
  extension uses Apple's Accelerate framework, so nothing else is needed.
- **Linux**: a compiler and the OpenBLAS headers, e.g. on Debian/Ubuntu
  `sudo apt-get install build-essential libopenblas-dev`.

There is no pure-Ruby fallback: if the extension can't be built, Maurograd
can't be loaded.

For the Apple Silicon GPU backend, install mlx-c first, then the separate
`maurograd-mlx` gem:

```sh
brew install mlx-c
gem install maurograd-mlx
```

To work on Maurograd itself: `bundle install`, then `./bin/build_ext` to
build the extensions (the MLX one is skipped without mlx-c) and
`bundle exec bake ci` to build and run the tests.


## Examples

While a full documentation will come down the road, Maurograd comes with a
couple of examples under `examples/` that should explain how to use the
framework:

- [`mnist.rb`](examples/mnist.rb): a multi-class classifier for digit
  recognition on the MNIST dataset, based on the LeNet CNN architecture.
- [`mnist_batchnorm.rb`](examples/mnist_batchnorm.rb): the same LeNet-style
  CNN, trained twice - once as-is, once with a `BatchNorm2D` layer after
  every `Conv2D` - so you can see what BatchNorm actually changes about
  training, rather than just reading about it.

Models with more than a chain of layers subclass `Layers::Module`: keep
layers and parameters in instance variables and write `forward`, and
`parameters`, `train`/`eval` and `to` work with no further code. A
residual block, for instance:

```ruby
class ResidualBlock < Maurograd::Layers::Module
  def initialize(channels)
    @conv1 = Maurograd::Layers::Conv2D.new(channels, channels, 3, padding: 1)
    @conv2 = Maurograd::Layers::Conv2D.new(channels, channels, 3, padding: 1)
  end

  def forward(x)
    x + @conv2.call(@conv1.call(x).relu)
  end
end
```

To save a trained model and load it back later, use its `state_dict`,
written in the [safetensors](https://github.com/huggingface/safetensors)
format (readable from Python too):

```ruby
Maurograd::SafeTensors.save(model.state_dict, "model.safetensors")
model.load_state_dict(Maurograd::SafeTensors.load("model.safetensors"))
```

Wrap evaluation and sampling in `Maurograd.no_grad { ... }`, so that no
graph is recorded, and call `model.eval` first, so that Dropout and
BatchNorm behave as at inference time.

Besides arithmetic and the layers, tensors have differentiable elementwise
functions (`exp`, `log`, `sigmoid`, `tanh`, `silu`, `softplus`,
`log_sigmoid`, `leaky_relu`, `clamp`), reductions (`sum` and `mean` over
any axes, `logsumexp`, `log_softmax`, `softmax`), and ways to cut and join
them (`slice`, `chunk`, `Maurograd.concat`). `Maurograd.randn` and
`Maurograd.rand` draw random tensors on their backend, and `detach` cuts a
tensor off the graph. For instance, a VAE's reparameterization trick:

```ruby
mean, log_var = encoder.call(x).chunk(2, axis: 1)
z = mean + (log_var * 0.5).exp * Maurograd.randn_like(mean)
```

(An MLP classifier for the two moons dataset also exists, but it's built on
the scalar-valued engine, so it now lives in
[maurograd_scalar](https://github.com/mtortonesi/maurograd_scalar) instead.)


## Backends

The autograd engine is the same everywhere, but tensor storage and the math
run on a pluggable *backend* (`lib/maurograd/backends/`):

- `:cpu` (default): Numo arrays, plus the C kernels in `ext/maurograd_ext`.
  The C extension links Accelerate on macOS and OpenBLAS elsewhere; set
  `MAUROGRAD_BLAS=openblas` (or `accelerate`) when building to choose.
- `:mlx`: [MLX](https://github.com/ml-explore/mlx) arrays, running on the
  Apple Silicon GPU, through bindings for
  [mlx-c](https://github.com/ml-explore/mlx-c) in `ext/maurograd_mlx`. It
  ships as the separate `maurograd-mlx` gem (see Installation); once that
  is installed, Maurograd loads it by itself when you ask for `:mlx`.

Pick one for a whole script with `MAUROGRAD_BACKEND=mlx`, or in code with
`Maurograd.backend = :mlx` (before building the model) or `model.to(:mlx)`.
`tensor.item` and `tensor.to_numo` read values back on any backend.

Two runtime defaults worth knowing about:

- BLAS/OpenMP thread pools default to a single thread
  (`OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`, unless you set them):
  numo-linalg-alt's bundled OpenBLAS keeps busy-waiting threads that slow
  training down, badly so on Linux, where the extension's own OpenBLAS adds
  a second pool.
- The MLX backend runs float32 matrix products at full precision
  (`MLX_ENABLE_TF32=0`, unless you set it), like the CPU backend.



## Disclaimers

At its heart, Maurograd is a didactic project. It is designed to be as simple
as possible, while reasonably CPU-competent. The project's goal is to enable
students to easily understand the inner working of serious ANN framework, by
leveraging Ruby's expressiveness and readability. So, Maurograd is not a
state-of-the-art ANN solution from the performance point of view - and never
will be. If you have a performance-oriented focus, you will be much better
server by [torch.rb](https://github.com/ankane/torch.rb) or (if you are on
Apple silicon) [mlx-ruby](https://github.com/skryl/mlx-ruby).

In addition, while developing Maurograd I heavily used AI-assisted coding
solutions. A secondary goal of this project for me was to familiarize with
state-of-the-art AI coding tools. I am completely open and unapologetic about
this. Of course, I ended up revising most of the code generated by AI. I would
guess that at the time of this writing (2026) the majority of software project
are being developed with a similar approach.


## License

This software is available as open source under the terms of the [MIT
License](https://opensource.org/licenses/MIT).
