require "maurograd/utils/utils"

describe "Maurograd::Utils.rand_norm" do
  it "produces identical arrays for the same seed" do
    a = Maurograd::Utils.rand_norm(4, 5, mean: 0.0, std: 1.0, seed: 42)
    b = Maurograd::Utils.rand_norm(4, 5, mean: 0.0, std: 1.0, seed: 42)
    expect(a).to be == b
  end

  it "produces different arrays for different seeds" do
    a = Maurograd::Utils.rand_norm(4, 5, mean: 0.0, std: 1.0, seed: 1)
    b = Maurograd::Utils.rand_norm(4, 5, mean: 0.0, std: 1.0, seed: 2)
    expect(a.ne(b).any?).to be == true
  end

  it "returns the requested shape" do
    a = Maurograd::Utils.rand_norm(3, 2, 5, seed: 0)
    expect(a.shape).to be == [3, 2, 5]
  end

  it "approximates the requested mean and standard deviation" do
    a = Maurograd::Utils.rand_norm(200_000, mean: 3.0, std: 2.0, seed: 7)
    mean = a.mean
    variance = ((a - mean)**2).mean
    expect((mean - 3.0).abs).to be < 0.05
    expect((Math.sqrt(variance) - 2.0).abs).to be < 0.05
  end

  it "falls back to Numo's own rand_norm when no seed is given" do
    a = Maurograd::Utils.rand_norm(3, 3, mean: 0.0, std: 1.0)
    expect(a.shape).to be == [3, 3]
  end
end
