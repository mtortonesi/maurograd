require 'maurograd/layers/conv2d'
require 'maurograd/layers/linear'

describe "Deterministic weight initialization via seed:" do
  it "gives Conv2D identical weights for the same seed" do
    a = Maurograd::Layers::Conv2D.new(2, 3, 3, seed: 42)
    b = Maurograd::Layers::Conv2D.new(2, 3, 3, seed: 42)
    expect(a.weights.data).to be == b.weights.data
  end

  it "gives Conv2D different weights for different seeds" do
    a = Maurograd::Layers::Conv2D.new(2, 3, 3, seed: 1)
    b = Maurograd::Layers::Conv2D.new(2, 3, 3, seed: 2)
    expect(a.weights.data.ne(b.weights.data).any?).to be == true
  end

  it "gives Linear identical weights for the same seed" do
    a = Maurograd::Layers::Linear.new(10, 5, seed: 7)
    b = Maurograd::Layers::Linear.new(10, 5, seed: 7)
    expect(a.weights.data).to be == b.weights.data
  end

  it "still constructs correctly when no seed is given" do
    conv = Maurograd::Layers::Conv2D.new(2, 3, 3)
    linear = Maurograd::Layers::Linear.new(10, 5)
    expect(conv.weights.shape).to be == [3, 2, 3, 3]
    expect(linear.weights.shape).to be == [5, 10]
  end
end
