require 'maurograd/tensor'
require 'maurograd/layers/conv2d'
require 'maurograd/layers/linear'

describe "Maurograd.seed!" do
  it "makes unseeded Conv2D weight init reproducible" do
    Maurograd.seed!(42)
    a = Maurograd::Layers::Conv2D.new(2, 3, 3)

    Maurograd.seed!(42)
    b = Maurograd::Layers::Conv2D.new(2, 3, 3)

    expect(a.weights.data).to be == b.weights.data
  end

  it "makes unseeded Linear weight init reproducible" do
    Maurograd.seed!(7)
    a = Maurograd::Layers::Linear.new(10, 5)

    Maurograd.seed!(7)
    b = Maurograd::Layers::Linear.new(10, 5)

    expect(a.weights.data).to be == b.weights.data
  end

  it "reproduces the same sequence across multiple layers built in order" do
    Maurograd.seed!(3)
    a1 = Maurograd::Layers::Conv2D.new(1, 4, 3)
    a2 = Maurograd::Layers::Linear.new(10, 2)

    Maurograd.seed!(3)
    b1 = Maurograd::Layers::Conv2D.new(1, 4, 3)
    b2 = Maurograd::Layers::Linear.new(10, 2)

    expect(a1.weights.data).to be == b1.weights.data
    expect(a2.weights.data).to be == b2.weights.data
  end

  it "gives different seeds different weights" do
    Maurograd.seed!(1)
    a = Maurograd::Layers::Conv2D.new(2, 3, 3)

    Maurograd.seed!(2)
    b = Maurograd::Layers::Conv2D.new(2, 3, 3)

    expect(a.weights.data.ne(b.weights.data).any?).to be == true
  end

  it "does not affect Datasets::Batching, which has its own independent seed" do
    require 'maurograd/datasets/batching'
    x = Numo::SFloat.new(6, 1, 2, 2).seq
    y = Numo::Int32.new(6).seq

    Maurograd.seed!(1)
    a = Maurograd::Datasets::Batching.batches(x, y, batch_size: 3).to_a.flat_map { |_, by| by.to_a }

    Maurograd.seed!(2)
    b = Maurograd::Datasets::Batching.batches(x, y, batch_size: 3).to_a.flat_map { |_, by| by.to_a }

    expect(a).to be == b
  end
end
