require "maurograd"

describe Maurograd::Layers::Dropout do
  def input(shape, requires_grad: true)
    Maurograd::Tensor.new(Numo::SFloat.new(*shape).rand + 0.5, requires_grad: requires_grad)
  end

  it "zeros about a fraction p of the elements and scales the rest by 1/(1-p)" do
    Maurograd.seed!(42)
    p = 0.3
    x = input([200, 100])
    y = Maurograd::Layers::Dropout.new(p).forward(x)

    dropped = y.data.eq(0).count.to_f / y.size
    expect((dropped - p).abs).to be < 0.02

    kept = y.data.ne(0)
    ratio = y.data[kept] / x.data[kept]
    expect((ratio - (1.0 / (1.0 - p))).abs.max).to be < 1e-5
  end

  it "preserves the mean in expectation" do
    Maurograd.seed!(7)
    x = Maurograd::Tensor.new(Numo::SFloat.ones(100_000))
    y = Maurograd::Layers::Dropout.new(0.5).forward(x)
    expect((y.data.mean - 1.0).abs).to be < 0.02
  end

  it "backpropagates through the same mask used in forward" do
    Maurograd.seed!(1)
    p = 0.5
    x = input([4, 5])
    y = Maurograd::Layers::Dropout.new(p).forward(x)
    y.sum.backward

    # d(sum(x * mask))/dx = mask, and mask = y / x elementwise.
    expect((x.grad - (y.data / x.data)).abs.max).to be < 1e-5
    expect(x.grad.eq(0).to_a).to be == y.data.eq(0).to_a
  end

  it "is the identity in eval mode" do
    x = input([3, 4])
    layer = Maurograd::Layers::Dropout.new(0.9).eval
    expect(layer.training?).to be == false
    expect(layer.forward(x).data).to be == x.data
  end

  it "is the identity with p = 0, and zeros everything with p = 1" do
    x = input([3, 4])
    expect(Maurograd::Layers::Dropout.new(0.0).forward(x).data).to be == x.data

    y = Maurograd::Layers::Dropout.new(1.0).forward(x)
    expect(y.data).to be == Numo::SFloat.zeros(3, 4)
    y.sum.backward
    expect(x.grad).to be == Numo::SFloat.zeros(3, 4)
  end

  it "switches mode through Sequential#train/#eval" do
    x = input([50, 50], requires_grad: false)
    model = Maurograd::Layers::Sequential.new(Maurograd::Layers::Dropout.new(0.5))

    model.eval
    expect(model.forward(x).data).to be == x.data

    model.train
    expect(model.forward(x).data.eq(0).count).to be > 0
  end

  it "draws reproducible masks under Maurograd.seed!" do
    x = input([10, 10])
    layer = Maurograd::Layers::Dropout.new(0.5)

    Maurograd.seed!(123)
    a = layer.forward(x).data
    Maurograd.seed!(123)
    b = layer.forward(x).data
    expect(a).to be == b
  end

  it "rejects probabilities outside [0, 1]" do
    expect { Maurograd::Layers::Dropout.new(-0.1) }.to raise_exception(ArgumentError)
    expect { Maurograd::Layers::Dropout.new(1.5) }.to raise_exception(ArgumentError)
    expect { Maurograd::Ops::Dropout.apply(input([2]), p: 2.0) }.to raise_exception(ArgumentError)
  end
end
