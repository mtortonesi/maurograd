require_relative 'backpropagate.rb'
require_relative 'value.rb'

class Neuron
  attr_accessor :weights, :bias

  def initialize(input_size:,nonlinear: true)
    @weights = Array.new(input_size) { Value.new(data: rand(-1.0..1.0)) }
    @bias = Value.new(data: rand(-1.0..1.0))
    @nonlinear = nonlinear
  end

  def call(inputs:)
    raise "Input size (#{inputs.size}) does not match weights size (#{@weights.size})" if inputs.size != @weights.size
    out = @weights.zip(inputs).map { |w, i| w * i }.reduce { |a, b| a + b } + @bias
    @nonlinear ? out.tanh : out
  end

  def parameters
    @weights + [ @bias ]
  end
end
