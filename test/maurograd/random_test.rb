require "maurograd"

describe "Maurograd.randn and Maurograd.rand" do
  it "draws tensors of the requested shape on the default backend" do
    x = Maurograd.randn(3, 4)
    expect(x.shape).to be == [3, 4]
    expect(x.data).to be_a(Numo::SFloat)
    expect(x.requires_grad).to be == false
    expect(Maurograd.rand([2, 5]).shape).to be == [2, 5]
    expect(Maurograd.randn(4, requires_grad: true).requires_grad).to be == true
  end

  it "has the right distributions" do
    Maurograd.seed!(1)
    n = Maurograd.randn(20_000).data
    u = Maurograd.rand(20_000).data
    expect(n.mean.abs).to be < 0.03
    expect((n.stddev - 1).abs).to be < 0.03
    expect(u.min).to be >= 0
    expect(u.max).to be < 1
    expect((u.mean - 0.5).abs).to be < 0.01
  end

  it "is reproducible with Maurograd.seed!" do
    Maurograd.seed!(7)
    a = Maurograd.randn(5).data
    b = Maurograd.rand(5).data
    Maurograd.seed!(7)
    expect(Maurograd.randn(5).data).to be == a
    expect(Maurograd.rand(5).data).to be == b
  end

  it "follows another tensor's shape and backend with randn_like and rand_like" do
    x = Maurograd::Tensor.new(Numo::SFloat.zeros(2, 3), requires_grad: true)
    eps = Maurograd.randn_like(x)
    expect(eps.shape).to be == [2, 3]
    expect(eps.backend).to be == x.backend
    expect(eps.requires_grad).to be == false
    expect(Maurograd.rand_like(x).shape).to be == [2, 3]
  end

  it "rejects invalid shapes" do
    expect { Maurograd.randn(2, -1) }.to raise_exception(ArgumentError)
    expect { Maurograd.rand(2.5) }.to raise_exception(ArgumentError)
  end
end
