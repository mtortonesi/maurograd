require 'maurograd/tensor'
require 'maurograd/ops/sub'

describe Maurograd::Ops::Sub do
  it "computes forward subtraction" do
    a = Maurograd::Tensor.new(Numo::SFloat[5, 7, 9], requires_grad: true)
    b = Maurograd::Tensor.new(Numo::SFloat[1, 2, 3], requires_grad: true)

    c = Maurograd::Ops::Sub.apply(a, b)
    expect(c.data).to be == Numo::SFloat[4, 5, 6]
  end

  it "computes backward gradients (no broadcasting)" do
    a = Maurograd::Tensor.new(Numo::SFloat[5, 7, 9], requires_grad: true)
    b = Maurograd::Tensor.new(Numo::SFloat[1, 2, 3], requires_grad: true)

    c = Maurograd::Ops::Sub.apply(a, b)
    c.backward(Numo::SFloat[1, 1, 1])

    # d(a-b)/da = 1
    expect(a.grad).to be == Numo::SFloat[1, 1, 1]
    # d(a-b)/db = -1
    expect(b.grad).to be == Numo::SFloat[-1, -1, -1]
  end

  it "handles broadcasting in backward (matrix - row-bias style)" do
    a = Maurograd::Tensor.new(Numo::SFloat[[1, 2, 3],
                                           [4, 5, 6]], requires_grad: true)
    b = Maurograd::Tensor.new(Numo::SFloat[[10, 20, 30]], requires_grad: true)

    c = Maurograd::Ops::Sub.apply(a, b)
    expect(c.data).to be == Numo::SFloat[[-9, -18, -27],
                                         [-6, -15, -24]]

    go = Numo::SFloat.ones(2, 3)
    c.backward(go)

    # grad_a = ones
    expect(a.grad).to be == Numo::SFloat.ones(2, 3)

    # grad_b = sum(-ones) over axis 0, keepdims -> [-2, -2, -2]
    expect(b.grad.shape).to be == [1, 3]
    expect(b.grad).to be == Numo::SFloat[[-2, -2, -2]]
  end
end

