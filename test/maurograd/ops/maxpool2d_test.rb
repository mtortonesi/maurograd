require "maurograd/tensor"
require "maurograd/ops/maxpool2d"

describe Maurograd::Ops::MaxPool2D do
  it "computes correct forward for a simple 2x2 -> 1x1 pool" do
    x = Maurograd::Tensor.new(Numo::SFloat[[[[1, 2],
      [3, 4]]]], requires_grad: false) # [1,1,2,2]

    y = Maurograd::Ops::MaxPool2D.apply(x, 2, 2, 2, 0)
    expect(y.shape).to be == [1, 1, 1, 1]
    expect(y.data).to be == Numo::SFloat[[[[4]]]]
  end

  it "backpropagates gradient only to max locations (no ties)" do
    x = Maurograd::Tensor.new(Numo::SFloat[[[[1, 2],
      [9, 4]]]], requires_grad: true) # max is 9 at [1,0]

    y = Maurograd::Ops::MaxPool2D.apply(x, 2, 2, 2, 0)

    # upstream gradient = 1
    y.backward(Numo::SFloat[[[[1]]]])

    expected = Numo::SFloat[[[[0, 0],
      [1, 0]]]]
    expect(x.grad).to be == expected
  end

  it "supports multiple channels independently" do
    x = Maurograd::Tensor.new(
      Numo::SFloat[
        [[[1, 5],
          [2, 3]],
          [[7, 1],
            [0, 4]]]
      ].reshape(1, 2, 2, 2),
      requires_grad: false
    )

    # kernel 2, stride 2 -> output [1,2,1,1] with per-channel maxes: [5] and [7]
    y = Maurograd::Ops::MaxPool2D.apply(x, 2, 2, 2, 0)
    expect(y.shape).to be == [1, 2, 1, 1]
    expect(y.data[0, 0, 0, 0]).to be == 5.0
    expect(y.data[0, 1, 0, 0]).to be == 7.0
  end

  it "backpropagates correctly with padding (gradient flows to max position in padded window)" do
    # Input [1,1,2,2]:
    # 9 2
    # 3 4
    x = Maurograd::Tensor.new(
      Numo::SFloat[[[[9, 2],
        [3, 4]]]],
      requires_grad: true
    )

    y = Maurograd::Ops::MaxPool2D.apply(x, 2, 2, 2, 1)
    expect(y.shape).to be == [1, 1, 2, 2]

    # Upstream gradient: only top-left output gets gradient 1
    dy = Numo::SFloat.zeros(1, 1, 2, 2)
    dy[0, 0, 0, 0] = 1.0
    y.backward(dy)

    # The top-left pooling window is:
    # [[0,0],
    #  [0,9]]  (in padded coords)
    # max is 9 -> corresponds to x[0,0].
    expected_dx = Numo::SFloat[[[[1, 0],
      [0, 0]]]]
    expect(x.grad).to be == expected_dx
  end
end
