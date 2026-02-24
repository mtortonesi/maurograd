# test/maurograd/scalar/value_ops_test.rb
require 'maurograd/scalar'

describe Maurograd::Scalar::Value do
  it "computes forward values for +, *, **" do
    x = Maurograd::Scalar::Value.new(data: 2.0)
    y = Maurograd::Scalar::Value.new(data: 3.0)

    z = (x + y) * (x ** 2) # (2+3) * 4 = 20
    expect(z.data).to be == 20.0
  end

  it "backpropagates correct gradients for a simple expression" do
    # f(x) = (x^2 + x) at x=3
    # df/dx = 2x + 1 = 7
    x = Maurograd::Scalar::Value.new(data: 3.0)
    f = (x ** 2) + x

    f.backpropagate
    expect(x.gradient).to be_within(1e-12).of(7.0)
  end

  it "accumulates gradient when a Value is used more than once (shared node)" do
    # f(x) = x*x + x  at x=4
    # df/dx = 2x + 1 = 9
    x = Maurograd::Scalar::Value.new(data: 4.0)
    f = (x * x) + x

    f.backpropagate
    expect(x.gradient).to be_within(1e-12).of(9.0)
  end

  it "supports Ruby Numeric on the left via coerce" do
    x = Maurograd::Scalar::Value.new(data: 2.0)
    y = 3.0 + x
    expect(y.data).to be == 5.0

    y.backpropagate
    expect(x.gradient).to be_within(1e-12).of(1.0)
  end
end

