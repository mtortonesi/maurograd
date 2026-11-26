require "maurograd"

describe Maurograd::Backends do
  it "fetches the CPU backend by name and passes backend objects through" do
    cpu = Maurograd::Backends.fetch(:cpu)
    expect(cpu.name).to be == :cpu
    expect(Maurograd::Backends.fetch("cpu")).to be_equal(cpu)
    expect(Maurograd::Backends.fetch(cpu)).to be_equal(cpu)
  end

  it "rejects unknown backends" do
    expect { Maurograd::Backends.fetch(:nope) }.to raise_exception(ArgumentError)
  end

  it "defaults tensors to Maurograd.backend" do
    t = Maurograd::Tensor.new([1.0, 2.0])
    expect(t.backend).to be_equal(Maurograd.backend)
    expect(t.device).to be == Maurograd.backend.name
  end

  it "exposes item and to_numo" do
    t = Maurograd::Tensor.new([[1.0, 2.0], [3.0, 4.0]])
    expect(t.to_numo).to be == Numo::SFloat[[1, 2], [3, 4]]
    expect(t.sum.item).to be == 10.0
    expect { t.item }.to raise_exception(ArgumentError)
  end

  it "keeps a tensor in place when moved to its own backend" do
    t = Maurograd::Tensor.new([1.0], requires_grad: true)
    data = t.data
    expect(t.to(:cpu)).to be_equal(t)
    expect(t.data).to be_equal(data)
  end

  it "refuses ops that mix backends" do
    # A second backend object with CPU semantics but a different identity.
    fake = Object.new
    Maurograd::Backends::CPU.singleton_methods.each do |m|
      fake.define_singleton_method(m) { |*a, **k, &b| Maurograd::Backends::CPU.public_send(m, *a, **k, &b) }
    end
    fake.define_singleton_method(:name) { :fake }

    a = Maurograd::Tensor.new([1.0])
    b = Maurograd::Tensor.new([2.0], backend: fake)
    expect { a + b }.to raise_exception(ArgumentError, message: be =~ /different backends/)
  end

  it "one-hot encodes class indices and rejects out-of-range ones" do
    cpu = Maurograd::Backends::CPU
    expect(cpu.one_hot(Numo::Int32[2, 0], 3)).to be == Numo::SFloat[[0, 0, 1], [1, 0, 0]]
    expect { cpu.one_hot(Numo::Int32[3], 3) }.to raise_exception(ArgumentError)
  end
end

describe "Conv2D backward" do
  it "computes only the gradients that are needed" do
    x = Maurograd::Tensor.new(Numo::SFloat.new(2, 1, 6, 6).rand, requires_grad: false)
    conv = Maurograd::Layers::Conv2D.new(1, 2, 3)
    conv.forward(x).sum.backward

    expect(x.grad).to be_nil
    expect(conv.weights.grad).not.to be_nil
    expect(conv.bias.grad).not.to be_nil

    # And at the extension level: false flags yield nil gradients.
    y, ctx = Maurograd::Backends::CPU.conv2d_forward(x.data, conv.weights.data, nil, 1, 0)
    dx, dw, db = Maurograd::Backends::CPU.conv2d_backward(
      ctx, Numo::SFloat.ones(*y.shape), want_dx: false, want_dw: true, want_db: false
    )
    expect(dx).to be_nil
    expect(db).to be_nil
    expect(dw.shape).to be == [2, 1, 3, 3]
  end
end
