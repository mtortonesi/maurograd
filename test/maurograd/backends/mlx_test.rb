require "maurograd"

# Cross-backend tests: every op runs on the CPU backend (the reference) and
# on the MLX backend from identical inputs, and forward outputs plus
# gradients must agree. Skipped where the MLX backend can't be loaded.
describe "MLX backend" do
  def mlx_available?
    Maurograd::Backends.fetch(:mlx)
    true
  rescue LoadError
    false
  end

  # Skip every test here where the MLX backend can't be loaded (e.g. on CI,
  # which has no mlx-c). Done in `around`, not `before`: sus runs `before`
  # outside the catch(:skip) that wraps the test body, so calling skip there
  # escapes as an UncaughtThrowError.
  def around(&block)
    return super if mlx_available?

    @__assertions__.skip("MLX backend not available")
  end

  # Largest absolute difference relative to the reference's magnitude.
  def rel_err(got, want)
    got = Numo::DFloat.cast(got)
    want = Numo::DFloat.cast(want)
    expect(got.shape).to be == want.shape
    ((got - want).abs.max / [want.abs.max, 1e-6].max).to_f
  end

  # Runs `block` once per backend on tensors built from the same host
  # arrays. The block gets the input tensors and returns the output tensor;
  # we backprop a fixed random upstream gradient and collect, per backend,
  # [output, *input gradients] as Numo arrays.
  def on_both(*inputs, seed: 5)
    %i[cpu mlx].map do |b|
      ts = inputs.map { |x| Maurograd::Tensor.new(x, requires_grad: true, backend: b) }
      out = yield(*ts)
      Maurograd.seed!(seed) # same upstream gradient for both backends
      dy = (out.size > 1) ? Numo::SFloat.new(*out.shape).rand_norm : nil
      out.backward(dy)
      [out.to_numo, *ts.map { |t| grad_of(t) }]
    end
  end

  # Gradient of a tensor as a Numo array.
  def grad_of(t)
    t.backend.to_numo(t.grad)
  end

  def expect_match(cpu, mlx, tol: 1e-4)
    cpu.zip(mlx).each { |c, m| expect(rel_err(m, c)).to be < tol }
  end

  def randn(*shape)
    Numo::SFloat.new(*shape).rand_norm
  end

  it "keeps tensors on MLX and converts back to Numo" do
    t = Maurograd::Tensor.new(Numo::SFloat[[1, 2], [3, 4]], backend: :mlx)
    expect(t.data).to be_a(Maurograd::MLX::Array)
    expect(t.device).to be == :mlx
    expect(t.to_numo).to be == Numo::SFloat[[1, 2], [3, 4]]
    expect(t.sum.item).to be == 10.0

    t.to(:cpu)
    expect(t.data).to be_a(Numo::SFloat)
    t.to(:mlx)
    expect(t.data).to be_a(Maurograd::MLX::Array)
  end

  it "refuses to mix CPU and MLX tensors in one op" do
    a = Maurograd::Tensor.new([1.0], backend: :cpu)
    b = Maurograd::Tensor.new([1.0], backend: :mlx)
    expect { a + b }.to raise_exception(ArgumentError, message: be =~ /different backends/)
  end

  it "matches CPU on broadcasting arithmetic" do
    Maurograd.seed!(1)
    cpu, mlx = on_both(randn(2, 3, 4, 4), randn(1, 3, 1, 1)) { |a, b| ((a + b) * b - a)**2 }
    expect_match(cpu, mlx)
  end

  it "matches CPU on Linear" do
    Maurograd.seed!(2)
    cpu, mlx = on_both(randn(8, 5), randn(3, 5), randn(3)) { |x, w, b| Maurograd::Ops::Linear.apply(x, w, b) }
    expect_match(cpu, mlx)
  end

  [[1, 0], [1, 2], [2, 1]].each do |stride, padding|
    it "matches CPU on Conv2D (stride #{stride}, padding #{padding})" do
      Maurograd.seed!(3)
      cpu, mlx = on_both(randn(2, 3, 9, 9), randn(4, 3, 3, 3), randn(4)) do |x, w, b|
        Maurograd::Ops::Conv2D.apply(x, w, b, stride, padding)
      end
      expect_match(cpu, mlx)
    end
  end

  [[2, 2, 2, 0], [3, 3, 2, 0], [3, 2, 1, 1]].each do |kh, kw, stride, padding|
    it "matches CPU on MaxPool2D (#{kh}x#{kw}, stride #{stride}, padding #{padding})" do
      Maurograd.seed!(4)
      cpu, mlx = on_both(randn(2, 3, 8, 7)) { |x| Maurograd::Ops::MaxPool2D.apply(x, kh, kw, stride, padding) }
      expect_match(cpu, mlx)
    end
  end

  it "matches CPU on MaxPool2D ties: the gradient goes to the first maximum" do
    # Constant regions (as in MNIST backgrounds) make many windows tie.
    x = (Numo::SFloat.new(1, 1, 8, 8).rand * 4).floor # values 0..3; N = C = 1 on purpose
    [[2, 2, 2, 0], [3, 3, 1, 1]].each do |kh, kw, stride, padding|
      Maurograd.seed!(14)
      cpu, mlx = on_both(x) { |t| Maurograd::Ops::MaxPool2D.apply(t, kh, kw, stride, padding) }
      expect_match(cpu, mlx)
    end
  end

  it "matches CPU on Permute and transpose" do
    Maurograd.seed!(15)
    cpu, mlx = on_both(randn(2, 3, 4, 5)) { |x| x.permute(0, 2, 3, 1).transpose(1, 3) }
    expect_match(cpu, mlx)
  end

  it "matches CPU on ReLU" do
    Maurograd.seed!(5)
    cpu, mlx = on_both(randn(4, 6)) { |x| Maurograd::Ops::ReLU.apply(x) }
    expect_match(cpu, mlx)
  end

  it "matches CPU on BatchNorm2D, including running statistics and eval mode" do
    Maurograd.seed!(6)
    x = randn(4, 3, 5, 5) * 2.0 + 1.0
    layers = %i[cpu mlx].map do |b|
      Maurograd.backend = b
      Maurograd::Layers::BatchNorm2D.new(3)
    ensure
      Maurograd.backend = :cpu
    end

    results = layers.map do |bn|
      bk = bn.gamma.backend.name
      xt = Maurograd::Tensor.new(x, requires_grad: true, backend: bk)
      y = bn.forward(xt)
      # A random upstream gradient: an affine function of x (say 0.5*x - 1)
      # would make dx exactly zero, since BatchNorm's training-mode backward
      # projects out the mean and x_hat directions, leaving only rounding
      # noise to compare.
      Maurograd.seed!(16)
      y.backward(Numo::SFloat.new(*x.shape).rand_norm)
      train = [y.to_numo, grad_of(xt), grad_of(bn.gamma), grad_of(bn.beta),
        bn.gamma.backend.to_numo(bn.running_mean), bn.gamma.backend.to_numo(bn.running_var)]

      bn.eval
      xe = Maurograd::Tensor.new(x, requires_grad: true, backend: bk)
      ye = bn.forward(xe)
      ye.backward(Numo::SFloat.ones(*x.shape))
      train + [ye.to_numo, grad_of(xe)]
    end

    expect_match(*results)
  end

  it "matches CPU on the losses" do
    Maurograd.seed!(7)
    logits = randn(6, 4)
    labels = Numo::Int32[0, 3, 1, 2, 3, 0]
    soft = Numo::SFloat.new(6, 4).rand.then { |p| p / p.sum(axis: 1, keepdims: true) }
    probs = Numo::SFloat.new(6, 4).rand * 0.9 + 0.05
    targets = Numo::SFloat.new(6, 4).rand

    {
      "cross-entropy (indices)" => ->(b) { [Maurograd::Tensor.new(logits, requires_grad: true, backend: b), Maurograd::Tensor.new(labels, backend: b), Maurograd::Losses::CrossEntropyWithLogits] },
      "cross-entropy (soft)" => ->(b) { [Maurograd::Tensor.new(logits, requires_grad: true, backend: b), Maurograd::Tensor.new(soft, backend: b), Maurograd::Losses::CrossEntropyWithLogits] },
      "MSE" => ->(b) { [Maurograd::Tensor.new(probs, requires_grad: true, backend: b), Maurograd::Tensor.new(targets, backend: b), Maurograd::Losses::MSE] },
      "BCE" => ->(b) { [Maurograd::Tensor.new(probs, requires_grad: true, backend: b), Maurograd::Tensor.new(targets.round, backend: b), Maurograd::Losses::BCE] }
    }.each_value do |make|
      cpu, mlx = %i[cpu mlx].map do |b|
        pred, target, loss_class = make.call(b)
        loss = loss_class.apply(pred, target)
        loss.backward
        [Numo::SFloat[loss.item], grad_of(pred)]
      end
      expect_match(cpu, mlx)
    end
  end

  it "rejects out-of-range class indices" do
    logits = Maurograd::Tensor.new(Numo::SFloat.zeros(2, 3), backend: :mlx)
    labels = Maurograd::Tensor.new(Numo::Int32[0, 3], backend: :mlx)
    expect { Maurograd::Losses::CrossEntropyWithLogits.apply(logits, labels) }.to raise_exception(ArgumentError)
  end

  it "drops about a fraction p of the elements in Dropout" do
    Maurograd.seed!(8)
    x = Maurograd::Tensor.new(Numo::SFloat.ones(200, 100), backend: :mlx)
    y = Maurograd::Layers::Dropout.new(0.3).forward(x).to_numo
    expect((y.eq(0).count.to_f / y.size - 0.3).abs).to be < 0.02
    expect((y[y.ne(0)] - 1.0 / 0.7).abs.max).to be < 1e-5
  end

  it "trains LeNet identically to CPU for a few steps (SGD with momentum, then Adam)" do
    Maurograd.seed!(9)
    x = Numo::SFloat.new(16, 1, 28, 28).rand
    y = Numo::Int32.new(16).rand(10)

    build = lambda do
      Maurograd.seed!(10) # same weights on both backends (initialized on the host)
      Maurograd::Layers::Sequential.new(
        Maurograd::Layers::Conv2D.new(1, 6, 5, padding: 2), Maurograd::Layers::BatchNorm2D.new(6),
        Maurograd::Layers::ReLU.new, Maurograd::Layers::MaxPool2D.new(2, stride: 2),
        Maurograd::Layers::Conv2D.new(6, 16, 5), Maurograd::Layers::ReLU.new,
        Maurograd::Layers::MaxPool2D.new(2, stride: 2), Maurograd::Layers::Flatten.new,
        Maurograd::Layers::Linear.new(400, 32), Maurograd::Layers::ReLU.new, Maurograd::Layers::Linear.new(32, 10)
      )
    end

    params = %i[cpu mlx].map do |b|
      model = build.call.to(b)
      [Maurograd::Optim::SGD.new(model.parameters, lr: 0.05, momentum: 0.9),
        Maurograd::Optim::Adam.new(model.parameters, lr: 1e-3)].each do |opt|
        2.times do
          opt.zero_grad
          logits = model.forward(Maurograd::Tensor.new(x, backend: b))
          Maurograd::Losses::CrossEntropyWithLogits.apply(logits, Maurograd::Tensor.new(y, backend: b)).backward
          opt.step
        end
      end
      # Leave out conv1's bias: it feeds BatchNorm2D, which subtracts the
      # per-channel mean, so its gradient is exactly zero in exact arithmetic
      # and both backends only compute rounding noise (~1e-7). Adam divides
      # each gradient by its own running RMS, turning that noise into
      # full-size steps of random sign - expected, not a backend mismatch.
      model.parameters.map(&:to_numo).reject.with_index { |_, i| i == 1 }
    end

    expect_match(*params, tol: 1e-3)
  end
end
