require "maurograd/tensor"
require "maurograd/ops/maxpool2d"

describe Maurograd::Ops::MaxPool2D do
  it "supports padding in forward (zeros are included in border windows)" do
    # Input [1,1,2,2]:
    # 1 2
    # 3 4
    x = Maurograd::Tensor.new(
      Numo::SFloat[[[[1, 2],
        [3, 4]]]],
      requires_grad: false
    )

    # padding=1 => padded is 4x4:
    # 0 0 0 0
    # 0 1 2 0
    # 0 3 4 0
    # 0 0 0 0
    #
    # kernel=2, stride=2 => windows:
    # (0,0): [[0,0],[0,1]] -> max=1
    # (0,1): [[0,0],[2,0]] -> max=2
    # (1,0): [[0,3],[0,0]] -> max=3
    # (1,1): [[4,0],[0,0]] -> max=4
    y = Maurograd::Ops::MaxPool2D.apply(x, 2, 2, 2, 1)

    expect(y.shape).to be == [1, 1, 2, 2]
    expect(y.data).to be == Numo::SFloat[[[[1, 2],
      [3, 4]]]]
  end
end
