# test/maurograd/scalar/value_activation_ops_test.rb
#
# Tests for the remaining Value operations not covered in value_ops_test.rb:
# tanh, relu, sigmoid, log, exp, subtraction, and zero_grad!.
require 'maurograd/scalar'

describe Maurograd::Scalar::Value do

  # -----------------------------------------------------------------------
  # tanh
  # -----------------------------------------------------------------------
  describe "#tanh" do
    it "computes the correct forward value" do
      x = Maurograd::Scalar::Value.new(data: 0.0)
      expect(x.tanh.data).to be_within(1e-12).of(0.0)

      x2 = Maurograd::Scalar::Value.new(data: 1.0)
      expect(x2.tanh.data).to be_within(1e-9).of(Math.tanh(1.0))
    end

    it "backpropagates the correct gradient: d/dx tanh(x) = 1 - tanh(x)^2" do
      x = Maurograd::Scalar::Value.new(data: 0.5)
      y = x.tanh
      y.backpropagate

      expected_grad = 1.0 - Math.tanh(0.5)**2
      expect(x.gradient).to be_within(1e-9).of(expected_grad)
    end
  end

  # -----------------------------------------------------------------------
  # relu
  # -----------------------------------------------------------------------
  describe "#relu" do
    it "passes through positive values unchanged" do
      x = Maurograd::Scalar::Value.new(data: 3.0)
      expect(x.relu.data).to be_within(1e-12).of(3.0)
    end

    it "clamps negative values to 0" do
      x = Maurograd::Scalar::Value.new(data: -2.0)
      expect(x.relu.data).to be_within(1e-12).of(0.0)
    end

    it "clamps zero to 0 (boundary)" do
      x = Maurograd::Scalar::Value.new(data: 0.0)
      expect(x.relu.data).to be_within(1e-12).of(0.0)
    end

    it "backpropagates gradient = 1 for positive inputs" do
      x = Maurograd::Scalar::Value.new(data: 2.0)
      y = x.relu
      y.backpropagate
      expect(x.gradient).to be_within(1e-12).of(1.0)
    end

    it "backpropagates gradient = 0 for negative inputs" do
      x = Maurograd::Scalar::Value.new(data: -1.0)
      y = x.relu
      y.backpropagate
      expect(x.gradient).to be_within(1e-12).of(0.0)
    end

    it "backpropagates gradient = 0 at the boundary (data == 0)" do
      x = Maurograd::Scalar::Value.new(data: 0.0)
      y = x.relu
      y.backpropagate
      expect(x.gradient).to be_within(1e-12).of(0.0)
    end
  end

  # -----------------------------------------------------------------------
  # sigmoid
  # -----------------------------------------------------------------------
  describe "#sigmoid" do
    it "computes the correct forward value" do
      x = Maurograd::Scalar::Value.new(data: 0.0)
      expect(x.sigmoid.data).to be_within(1e-9).of(0.5)

      x2 = Maurograd::Scalar::Value.new(data: 1.0)
      expected = 1.0 / (1.0 + Math.exp(-1.0))
      expect(x2.sigmoid.data).to be_within(1e-9).of(expected)
    end

    it "backpropagates the correct gradient: d/dx sigmoid(x) = s*(1-s)" do
      x = Maurograd::Scalar::Value.new(data: 2.0)
      y = x.sigmoid
      y.backpropagate

      s = 1.0 / (1.0 + Math.exp(-2.0))
      expected_grad = s * (1.0 - s)
      expect(x.gradient).to be_within(1e-9).of(expected_grad)
    end
  end

  # -----------------------------------------------------------------------
  # log
  # -----------------------------------------------------------------------
  describe "#log" do
    it "computes the correct forward value" do
      x = Maurograd::Scalar::Value.new(data: Math::E)
      expect(x.log.data).to be_within(1e-9).of(1.0)

      x2 = Maurograd::Scalar::Value.new(data: 1.0)
      expect(x2.log.data).to be_within(1e-12).of(0.0)
    end

    it "backpropagates the correct gradient: d/dx log(x) = 1/x" do
      x = Maurograd::Scalar::Value.new(data: 4.0)
      y = x.log
      y.backpropagate
      expect(x.gradient).to be_within(1e-9).of(1.0 / 4.0)
    end

    it "raises ArgumentError for zero" do
      x = Maurograd::Scalar::Value.new(data: 0.0)
      raised = false
      begin
        x.log
      rescue ArgumentError
        raised = true
      end
      expect(raised).to be == true
    end

    it "raises ArgumentError for negative values" do
      x = Maurograd::Scalar::Value.new(data: -1.0)
      raised = false
      begin
        x.log
      rescue ArgumentError
        raised = true
      end
      expect(raised).to be == true
    end
  end

  # -----------------------------------------------------------------------
  # exp
  # -----------------------------------------------------------------------
  describe "#exp" do
    it "computes the correct forward value" do
      x = Maurograd::Scalar::Value.new(data: 0.0)
      expect(x.exp.data).to be_within(1e-12).of(1.0)

      x2 = Maurograd::Scalar::Value.new(data: 1.0)
      expect(x2.exp.data).to be_within(1e-9).of(Math::E)
    end

    it "backpropagates the correct gradient: d/dx exp(x) = exp(x)" do
      x = Maurograd::Scalar::Value.new(data: 2.0)
      y = x.exp
      y.backpropagate
      expect(x.gradient).to be_within(1e-9).of(Math.exp(2.0))
    end
  end

  # -----------------------------------------------------------------------
  # - (subtraction)
  # -----------------------------------------------------------------------
  describe "#-" do
    it "computes the correct forward value" do
      x = Maurograd::Scalar::Value.new(data: 5.0)
      y = Maurograd::Scalar::Value.new(data: 3.0)
      expect((x - y).data).to be_within(1e-12).of(2.0)
    end

    it "works with a plain Numeric on the right" do
      x = Maurograd::Scalar::Value.new(data: 7.0)
      expect((x - 2.0).data).to be_within(1e-12).of(5.0)
    end

    it "backpropagates correct gradients: d(a-b)/da = 1, d(a-b)/db = -1" do
      a = Maurograd::Scalar::Value.new(data: 3.0)
      b = Maurograd::Scalar::Value.new(data: 1.0)
      c = a - b
      c.backpropagate

      expect(a.gradient).to be_within(1e-12).of(1.0)
      expect(b.gradient).to be_within(1e-12).of(-1.0)
    end
  end

  # -----------------------------------------------------------------------
  # zero_grad!
  # -----------------------------------------------------------------------
  describe "#zero_grad!" do
    it "resets all gradients in the computation graph to 0.0" do
      x = Maurograd::Scalar::Value.new(data: 2.0)
      y = Maurograd::Scalar::Value.new(data: 3.0)
      z = (x * y) + x   # x gradient should be non-zero after backprop

      z.backpropagate
      expect(x.gradient.abs > 1e-12).to be == true

      z.zero_grad!
      expect(x.gradient).to be_within(1e-12).of(0.0)
      expect(y.gradient).to be_within(1e-12).of(0.0)
    end
  end

end
