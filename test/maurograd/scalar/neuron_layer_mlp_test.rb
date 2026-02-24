# test/maurograd/scalar/neuron_layer_mlp_test.rb
require 'maurograd/scalar'

describe "Maurograd::Scalar network pieces" do
  it "Neuron forward returns a Value (and backprop works)" do
    n = Maurograd::Scalar::Neuron.new(input_size: 3, nonlinear: false)

    x1 = Maurograd::Scalar::Value.new(data: 1.0)
    x2 = Maurograd::Scalar::Value.new(data: -2.0)
    x3 = Maurograd::Scalar::Value.new(data: 0.5)

    out = n.call(inputs: [x1, x2, x3])
    expect(out).to be_a(Maurograd::Scalar::Value)

    out.backpropagate

    # We don't assert exact numbers (random weights),
    # but gradients must exist and be numeric.
    params = n.parameters
    expect(params.length).to be == 4 # 3 weights + bias
    params.each do |p|
      expect(p.gradient).to be_a(Numeric)
    end
  end

  it "Layer produces the right number of outputs" do
    layer = Maurograd::Scalar::Layer.new(input_size: 2, output_size: 3, nonlinear: false)

    x1 = Maurograd::Scalar::Value.new(data: 1.0)
    x2 = Maurograd::Scalar::Value.new(data: 2.0)

    out = layer.call(inputs: [x1, x2])
    expect(out).to be_a(Array)
    expect(out.length).to be == 3
    expect(out.all? { |v| v.is_a?(Maurograd::Scalar::Value) }).to be_truthy
  end

  it "MLP forward works end-to-end and gradients reach parameters" do
    # 2 -> 3 -> 1
    mlp = Maurograd::Scalar::MLP.new(layer_sizes: [2, 3, 1])

    x1 = Maurograd::Scalar::Value.new(data: 1.0)
    x2 = Maurograd::Scalar::Value.new(data: -1.0)

    out = mlp.call(inputs: [x1, x2])
    # last layer has output_size=1 so your code returns a single Value
    expect(out).to be_a(Maurograd::Scalar::Value)

    # Make a tiny scalar loss: L = out^2
    loss = out ** 2
    loss.backpropagate

    # Ensure some parameter got non-zero-ish gradient (not guaranteed for all, but likely).
    params = mlp.parameters
    expect(params.length).to be > 0
    expect(params.all? { |p| p.is_a?(Maurograd::Scalar::Value) }).to be_truthy

    # At least one gradient should be non-zero (very likely unless pathological random init).
    # We keep it lenient to avoid flaky tests.
    nonzero = params.any? { |p| p.gradient.abs > 1e-12 }
    expect(nonzero).to be_truthy
  end
  it "Neuron with nonlinear:true applies tanh and output is in (-1, 1)" do
    n = Maurograd::Scalar::Neuron.new(input_size: 2, nonlinear: true)

    x1 = Maurograd::Scalar::Value.new(data: 1.0)
    x2 = Maurograd::Scalar::Value.new(data: 1.0)

    out = n.call(inputs: [x1, x2])
    expect(out).to be_a(Maurograd::Scalar::Value)

    # tanh output is always in the open interval (-1, 1)
    expect(out.data).to be > -1.0
    expect(out.data).to be < 1.0

    out.backpropagate

    # All parameters should receive a numeric gradient
    n.parameters.each do |p|
      expect(p.gradient).to be_a(Numeric)
    end
  end
end

