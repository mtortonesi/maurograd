require "maurograd"

describe Maurograd::Layers::Module do
  # A residual block, written as a user would: instance variables and a
  # forward, nothing else.
  let(:block_class) do
    Class.new(Maurograd::Layers::Module) do
      def initialize(channels)
        @conv1 = Maurograd::Layers::Conv2D.new(channels, channels, 3, padding: 1)
        @bn = Maurograd::Layers::BatchNorm2D.new(channels)
        @conv2 = Maurograd::Layers::Conv2D.new(channels, channels, 3, padding: 1)
        @drop = Maurograd::Layers::Dropout.new(0.5)
      end

      def forward(x)
        x + @drop.call(@conv2.call(@bn.call(@conv1.call(x)).relu))
      end
    end
  end

  def input(*shape)
    Maurograd::Tensor.new(Numo::SFloat.new(*shape).rand_norm)
  end

  it "finds parameters in instance variables, in assignment order" do
    linear = Maurograd::Layers::Linear.new(3, 2)
    expect(linear.parameters).to be == [linear.weights, linear.bias]
    expect(linear.named_parameters.map(&:first)).to be == %w[weights bias]
  end

  it "leaves out a missing bias, and Tensors that don't require grad" do
    layer = Class.new(Maurograd::Layers::Module) do
      def initialize
        @weights = Maurograd::Tensor.new(Numo::SFloat.ones(2), requires_grad: true)
        @bias = nil
        @constant = Maurograd::Tensor.new(Numo::SFloat.ones(2))
        @scale = 3.0
      end
    end.new

    expect(layer.named_parameters.map(&:first)).to be == %w[weights]
  end

  it "searches child modules, Arrays and Hashes, with dotted names" do
    model = Class.new(Maurograd::Layers::Module) do
      def initialize
        @stem = Maurograd::Layers::Linear.new(4, 4)
        @blocks = [Maurograd::Layers::Linear.new(4, 4), Maurograd::Layers::ReLU.new, Maurograd::Layers::Linear.new(4, 4, bias: false)]
        @heads = {mean: Maurograd::Layers::Linear.new(4, 2), logvar: Maurograd::Layers::Linear.new(4, 2)}
      end
    end.new

    expect(model.named_parameters.map(&:first)).to be == %w[
      stem.weights stem.bias
      blocks.0.weights blocks.0.bias blocks.2.weights
      heads.mean.weights heads.mean.bias heads.logvar.weights heads.logvar.bias
    ]
    expect(model.children.length).to be == 6
    expect(model.modules.length).to be == 7
  end

  it "lists a shared parameter or module once" do
    shared = Maurograd::Layers::Linear.new(3, 3)
    model = Class.new(Maurograd::Layers::Module) do
      def initialize(shared)
        @a = shared
        @b = shared
        @tied = shared.weights
      end
    end.new(shared)

    expect(model.parameters).to be == [shared.weights, shared.bias]
    expect(model.modules.length).to be == 2
  end

  it "names Sequential's parameters by position, in the same order as before" do
    model = Maurograd::Layers::Sequential.new(Maurograd::Layers::Conv2D.new(1, 2, 3), Maurograd::Layers::ReLU.new, Maurograd::Layers::Flatten.new, Maurograd::Layers::Linear.new(8, 2))
    expect(model.named_parameters.map(&:first)).to be == %w[
      layers.0.weights layers.0.bias layers.3.weights layers.3.bias
    ]
  end

  it "sets train and eval mode on every descendant" do
    model = Maurograd::Layers::Sequential.new(block_class.new(2), Maurograd::Layers::Flatten.new)
    expect(model.modules.all?(&:training?)).to be == true

    model.eval
    expect(model.modules.none?(&:training?)).to be == true

    model.train
    expect(model.modules.all?(&:training?)).to be == true
  end

  it "uses eval mode in nested Dropout and BatchNorm2D" do
    Maurograd.seed!(1)
    block = block_class.new(2)
    x = input(4, 2, 5, 5)

    block.eval
    first = block.call(x).data
    expect(block.call(x).data).to be == first # no dropout, running statistics
  end

  it "calls forward from call" do
    linear = Maurograd::Layers::Linear.new(3, 2)
    x = input(4, 3)
    expect(linear.call(x).data).to be == linear.forward(x).data
    expect(linear.call(x).data).to be == linear.forward(x).data
  end

  it "trains a custom block: gradients reach every parameter" do
    Maurograd.seed!(2)
    block = block_class.new(2)
    x = input(4, 2, 5, 5)
    opt = Maurograd::Optim::SGD.new(block.parameters, lr: 0.1)
    before = block.parameters.map { |p| p.data.dup }

    (block.call(x) * block.call(x)).sum.backward
    opt.step

    expect(block.parameters.length).to be == 6
    block.parameters.zip(before).each do |p, old|
      expect(p.grad).not.to be_nil
      expect(p.data).not.to be == old
    end
  end

  it "summarizes itself without printing the weights" do
    model = Maurograd::Layers::Sequential.new(Maurograd::Layers::Linear.new(3, 2), Maurograd::Layers::ReLU.new)
    expect(model.inspect).to be == <<~TEXT.chomp
      Sequential()
        layers.0: Linear(weights: [2, 3], bias: [2])
        layers.1: ReLU()
    TEXT
  end

  it "raises when forward isn't implemented" do
    expect { Class.new(Maurograd::Layers::Module).new.call(input(1)) }.to raise_exception(NotImplementedError)
  end

  it "rejects Sequential layers that aren't Modules" do
    plain = Object.new
    def plain.forward(x) = x
    expect { Maurograd::Layers::Sequential.new(Maurograd::Layers::ReLU.new, plain) }.to raise_exception(TypeError)
  end
end
