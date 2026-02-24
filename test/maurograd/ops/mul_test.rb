require 'maurograd/tensor'
require 'maurograd/ops/mul'

describe Maurograd::Ops::Mul do
  it "computes forward elementwise multiplication" do
    a = Maurograd::Tensor.new(Numo::SFloat[1, 2, 3], requires_grad: true)
    b = Maurograd::Tensor.new(Numo::SFloat[4, 5, 6], requires_grad: true)

    c = Maurograd::Ops::Mul.apply(a, b)
    expect(c.data).to be == Numo::SFloat[4, 10, 18]
  end

  it "computes backward gradients (no broadcasting)" do
    a = Maurograd::Tensor.new(Numo::SFloat[1, 2, 3], requires_grad: true)
    b = Maurograd::Tensor.new(Numo::SFloat[4, 5, 6], requires_grad: true)

    c = Maurograd::Ops::Mul.apply(a, b)
    c.backward(Numo::SFloat[1, 1, 1])

    # d(a*b)/da = b
    expect(a.grad).to be == Numo::SFloat[4, 5, 6]
    # d(a*b)/db = a
    expect(b.grad).to be == Numo::SFloat[1, 2, 3]
  end

  it "handles broadcasting in backward (vector * row-bias style)" do
    # a: [2,3], b: [1,3] -> broadcast b over axis 0
    a = Maurograd::Tensor.new(Numo::SFloat[[1, 2, 3],
                                           [4, 5, 6]], requires_grad: true)
    b = Maurograd::Tensor.new(Numo::SFloat[[10, 20, 30]], requires_grad: true)

    c = Maurograd::Ops::Mul.apply(a, b)
    expect(c.shape).to be == [2, 3]
    expect(c.data).to be == Numo::SFloat[[10, 40, 90],
                                         [40, 100, 180]]

    # upstream gradient all ones
    go = Numo::SFloat.ones(2, 3)
    c.backward(go)

    # grad_a = go * b (broadcast) => rows equal to b
    expect(a.grad).to be == Numo::SFloat[[10, 20, 30],
                                         [10, 20, 30]]

    # grad_b = sum over batch of (go * a) along axis 0, keepdims -> [1,3]
    # => [1+4, 2+5, 3+6] = [5,7,9]
    expect(b.grad.shape).to be == [1, 3]
    expect(b.grad).to be == Numo::SFloat[[5, 7, 9]]
  end
end

