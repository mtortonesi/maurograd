require_relative 'layer'

class MLP
  attr_reader :layers

  def initialize(layer_sizes:)
    @layers = []
    @input_size = layer_sizes.first
    layer_sizes.each_cons(2).with_index do |(in_sz, out_sz), idx|
      is_last = (idx == layer_sizes.length - 2)
      @layers << Layer.new(input_size: in_sz, output_size: out_sz, nonlinear: !is_last)
    end
  end

  def parameters
    layers.map(&:parameters).flatten
  end

  def call(inputs:)
    raise ArgumentError.new("MLP expects #{@input_size} inputs. Got #{inputs.size}.") unless inputs.size == @input_size
    layers.each_with_index do |layer,i|
      inputs = layer.call(inputs: inputs)
    end
    inputs
  end
end

