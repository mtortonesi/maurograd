require "maurograd/tensor"
require "maurograd/ops/linear"

describe Maurograd::Ops::Linear do
  it "computes Y = X @ W^T + b with correct shape and values" do
    # X: [2,3]
    x = Maurograd::Tensor.new(Numo::SFloat[[1, 2, 3],
      [4, 5, 6]], requires_grad: false)

    # W: [4,3] (Out=4, In=3)
    w = Maurograd::Tensor.new(Numo::SFloat[[1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 1, 1]], requires_grad: false)

    # b: [4]
    b = Maurograd::Tensor.new(Numo::SFloat[0.5, 1.0, -1.0, 0.0], requires_grad: false)

    y = Maurograd::Ops::Linear.apply(x, w, b)

    expect(y.shape).to be == [2, 4]

    # For row [1,2,3]:
    # out0 = 1 + 0.5 = 1.5
    # out1 = 2 + 1.0 = 3.0
    # out2 = 3 - 1.0 = 2.0
    # out3 = (1+2+3) + 0 = 6
    exp0 = Numo::SFloat[1.5, 3.0, 2.0, 6.0]

    # For row [4,5,6]:
    exp1 = Numo::SFloat[4.5, 6.0, 5.0, 15.0]

    expect(y.data[0, true]).to be == exp0
    expect(y.data[1, true]).to be == exp1
  end
end
