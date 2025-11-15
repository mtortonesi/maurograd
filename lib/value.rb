require_relative 'backpropagate'

class Value
  attr_accessor :data, :backward, :gradient
  attr_reader :children

  include Backpropagation

  def initialize(data:, children: [], gradient: 0.0)
    @data = data
    @gradient = gradient
    @children = children
    @backward = -> {} # default is no-op, e.g. for leaf nodes
  end

  def coerce(other)
    # When an L-value is a Numeric (e.g., 0 + value), Ruby calls coerce.
    # We need to return [other_as_value, self] to let Ruby proceed with the
    # operation.
    [Value.new(data: other), self]
  end

  def +(other)
    other = Value.new(data: other) if other.is_a?(Numeric)
    a = self; b = other
    out = Value.new(data: a.data + b.data, children: [a, b])
    out.backward = -> {
      a.gradient += 1.0 * out.gradient
      b.gradient += 1.0 * out.gradient
    }
    out
  end

  def *(other)
    other = Value.new(data: other) if other.is_a?(Numeric)
    a = self; b = other
    out = Value.new(data: a.data * b.data, children: [a, b])
    out.backward = -> {
      a.gradient += b.data * out.gradient
      b.gradient += a.data * out.gradient
    }
    out
  end

  def **(other)
    raise ArgumentError, "Only implemented for scalar values" unless other.is_a?(Numeric)
    a = self
    out = Value.new(data: a.data ** other, children: [a])
    out.backward = -> {
      a.gradient += (other * (a.data ** (other - 1))) * out.gradient
    }
    out
  end

  def tanh
    a = self
    out = Value.new(data: Math.tanh(a.data), children: [a])
    out.backward = -> {
      a.gradient += (1.0 - out.data ** 2) * out.gradient
    }
    out
  end

  def relu
    a = self
    out = Value.new(data: a.data > 0 ? a.data : 0.0, children: [a])
    out.backward = -> {
      a.gradient += (out.data > 0 ? 1.0 : 0.0) * out.gradient
    }
    out
  end

  def -(other)
    self + (other * -1)
  end

  def log
    a = self
    raise ArgumentError, "log undefined for non-positive values" if a.data <= 0.0
    out = Value.new(data: Math.log(a.data), children: [a])
    out.backward = -> {
      a.gradient += (1.0 / a.data) * out.gradient
    }
    out
  end

  def sigmoid
    a = self
    s = 1.0 / (1.0 + Math.exp(-a.data))
    out = Value.new(data: s, children: [a])
    out.backward = -> {
      a.gradient += (s * (1.0 - s)) * out.gradient
    }
    out
  end

  def exp
    a = self
    e = Math.exp(a.data)
    out = Value.new(data: e, children: [a])
    out.backward = -> {
      a.gradient += e * out.gradient
    }
    out
  end
end
