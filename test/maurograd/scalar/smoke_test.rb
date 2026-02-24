# test/maurograd/scalar/smoke_test.rb
require 'maurograd/scalar'

describe Maurograd::Scalar do
  it "exposes the scalar classes under Maurograd::Scalar" do
    expect(defined?(Maurograd::Scalar::Value)).to be_truthy
    expect(defined?(Maurograd::Scalar::Neuron)).to be_truthy
    expect(defined?(Maurograd::Scalar::Layer)).to be_truthy
    expect(defined?(Maurograd::Scalar::MLP)).to be_truthy
  end

  it "does not leak global constants (Value/Neuron/Layer/MLP)" do
    # After refactor, these should NOT be defined at top level.
    expect(defined?(::Value)).to be_nil
    expect(defined?(::Neuron)).to be_nil
    expect(defined?(::Layer)).to be_nil
    expect(defined?(::MLP)).to be_nil
  end
end

