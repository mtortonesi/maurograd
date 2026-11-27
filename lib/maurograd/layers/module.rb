require_relative "../tensor"

module Maurograd
  module Layers
    # Base class for layers and models, like PyTorch's nn.Module.
    #
    # A subclass only implements #forward and keeps its state in instance
    # variables. Module finds that state by itself, by looking at the
    # instance variables:
    #
    # - a Tensor that requires grad is a parameter;
    # - a Module is a child (a sublayer), searched in turn;
    # - Arrays and Hashes are searched one level deep, so @blocks = [...]
    #   or @heads = {mean: ..., logvar: ...} work too.
    #
    # Everything else (numbers, flags, native arrays) is ignored. So a
    # residual block is just:
    #
    #   class ResidualBlock < Maurograd::Layers::Module
    #     def initialize(channels)
    #       @conv1 = Conv2D.new(channels, channels, 3, padding: 1)
    #       @conv2 = Conv2D.new(channels, channels, 3, padding: 1)
    #     end
    #
    #     def forward(x)
    #       x + @conv2.call(@conv1.call(x).relu)
    #     end
    #   end
    #
    # and #parameters, #train / #eval and #to work on it, and on any model
    # built from it, with no further code.
    #
    # The order is the order the instance variables were first assigned,
    # which Ruby preserves; optimizers rely on that order being stable. A
    # parameter or child reachable twice (weight tying, a shared block) is
    # listed once, so an optimizer updates it once.
    #
    # Subclasses don't need to call super in #initialize: a module that
    # never had #train or #eval called is in training mode.
    class Module
      # Calls #forward, so that model.(x) and model.call(x) work.
      def call(...)
        forward(...)
      end

      def forward(*)
        raise NotImplementedError, "#{self.class} must implement #forward"
      end

      # Trainable tensors of this module and all its descendants.
      def parameters
        named_parameters.map(&:last)
      end

      # [[name, tensor], ...], with dotted names that follow the instance
      # variables: "weights", "layers.0.bias", "blocks.1.conv1.weights".
      def named_parameters
        result = []
        collect_parameters("", {}.compare_by_identity, result)
        result
      end

      # Direct sublayers, as [[name, module], ...].
      def named_children
        each_member.select { |_, v| v.is_a?(Module) }
      end

      def children
        named_children.map(&:last)
      end

      # This module and all its descendants, each once.
      def modules
        result = []
        collect_modules({}.compare_by_identity, result)
        result
      end

      # Sets training mode (mode = true) or evaluation mode (mode = false)
      # on this module and all its descendants. Layers such as Dropout and
      # BatchNorm2D behave differently in the two modes.
      def train(mode = true)
        modules.each { |m| m.training = mode }
        self
      end

      def eval
        train(false)
      end

      def training?
        @training != false
      end

      # Moves every Tensor this module holds, and every descendant, to
      # another backend, e.g. model.to(:mlx). Layers with state that isn't
      # a Tensor (BatchNorm2D's running statistics) override this, call
      # super and move the rest themselves.
      def to(backend)
        each_member { |_, v| v.to(backend) }
        self
      end

      # A short summary: the class, its parameters' shapes, and its
      # children, indented. (The default #inspect would print every
      # weight.)
      def inspect
        own = each_member.select { |_, v| v.is_a?(Tensor) && v.requires_grad }
          .map { |name, t| "#{name}: #{t.shape.inspect}" }
        head = "#{self.class.name.sub(/\AMaurograd::Layers::/, "")}(#{own.join(", ")})"
        kids = named_children.map do |name, child|
          "  #{name}: " + child.inspect.gsub("\n", "\n  ")
        end
        kids.empty? ? head : ([head] + kids).join("\n")
      end

      protected

      attr_writer :training

      def collect_parameters(prefix, seen, result)
        each_member do |name, v|
          next if seen.key?(v)

          if v.is_a?(Tensor)
            next unless v.requires_grad

            seen[v] = true
            result << ["#{prefix}#{name}", v]
          else
            seen[v] = true
            v.collect_parameters("#{prefix}#{name}.", seen, result)
          end
        end
      end

      def collect_modules(seen, result)
        return if seen.key?(self)

        seen[self] = true
        result << self
        children.each { |c| c.collect_modules(seen, result) }
      end

      private

      # Yields (or, without a block, returns) [name, value] for every
      # Tensor and Module held in an instance variable, directly or inside
      # an Array or Hash.
      def each_member
        return enum_for(:each_member) unless block_given?

        instance_variables.each do |ivar|
          name = ivar.to_s.delete_prefix("@")
          value = instance_variable_get(ivar)
          case value
          when Tensor, Module
            yield name, value
          when Array
            value.each_with_index { |v, i| yield "#{name}.#{i}", v if member?(v) }
          when Hash
            value.each { |k, v| yield "#{name}.#{k}", v if member?(v) }
          end
        end
      end

      def member?(value)
        value.is_a?(Tensor) || value.is_a?(Module)
      end
    end
  end
end
