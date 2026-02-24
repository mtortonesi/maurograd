require 'numo/narray'
require 'maurograd/datasets/batching'

describe Maurograd::Datasets::Batching do
  it "yields mini-batches with correct shapes (no shuffle)" do
    x = Numo::SFloat.new(10, 1, 28, 28).seq.reshape(10, 1, 28, 28)
    y = Numo::Int32.new(10).seq

    batches = Maurograd::Datasets::Batching.batches(
      x, y,
      batch_size: 4,
      shuffle: false,
      drop_last: false
    ).to_a

    # 10 elems with bs=4 => 3 batches: 4,4,2
    expect(batches.length).to be == 3

    bx0, by0 = batches[0]
    expect(bx0.shape).to be == [4, 1, 28, 28]
    expect(by0.shape).to be == [4]

    bx2, by2 = batches[2]
    expect(bx2.shape).to be == [2, 1, 28, 28]
    expect(by2.shape).to be == [2]

    # order preserved
    expect(by0.to_a).to be == [0, 1, 2, 3]
    expect(by2.to_a).to be == [8, 9]
  end

  it "supports drop_last: true" do
    x = Numo::SFloat.new(10, 1, 28, 28).seq.reshape(10, 1, 28, 28)
    y = Numo::Int32.new(10).seq

    batches = Maurograd::Datasets::Batching.batches(
      x, y,
      batch_size: 4,
      shuffle: false,
      drop_last: true
    ).to_a

    # only full batches => 2 batches of 4
    expect(batches.length).to be == 2
    expect(batches[1][0].shape).to be == [4, 1, 28, 28]
    expect(batches[1][1].shape).to be == [4]
  end

  it "shuffles deterministically with seed" do
    x = Numo::SFloat.new(10, 1, 28, 28).seq.reshape(10, 1, 28, 28)
    y = Numo::Int32.new(10).seq

    b1 = Maurograd::Datasets::Batching.batches(x, y, batch_size: 5, shuffle: true, seed: 123).to_a
    b2 = Maurograd::Datasets::Batching.batches(x, y, batch_size: 5, shuffle: true, seed: 123).to_a

    # labels order must match exactly
    y1 = b1.flat_map { |_bx, by| by.to_a }
    y2 = b2.flat_map { |_bx, by| by.to_a }
    expect(y1).to be == y2

    # and different seed => (very likely) different order
    b3 = Maurograd::Datasets::Batching.batches(x, y, batch_size: 5, shuffle: true, seed: 999).to_a
    y3 = b3.flat_map { |_bx, by| by.to_a }
    expect(y3).not.to be == y1
  end
end
