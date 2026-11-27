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
    #
    # State that isn't a parameter but belongs to the model (BatchNorm2D's
    # running statistics) is declared as a buffer, by instance variable
    # name:
    #
    #   buffers :running_mean, :running_var
    #
    # Buffers hold native arrays. #to moves them, and #state_dict saves
    # them, along with the parameters.
    class Module
      # Declares buffers: instance variables, holding native arrays, that
      # #to moves and #state_dict includes. Inherited by subclasses.
      def self.buffers(*names)
        own_buffer_names.concat(names.map(&:to_s))
      end

      def self.buffer_names
        inherited = (superclass <= Module) ? superclass.buffer_names : []
        inherited + own_buffer_names
      end

      def self.own_buffer_names
        @own_buffer_names ||= []
      end
      private_class_method :own_buffer_names
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
        named_modules.map(&:last)
      end

      # [[name, module], ...] for this module ("") and all its descendants
      # ("layers.0", "blocks.1.conv1"), each once.
      def named_modules
        result = []
        collect_modules("", {}.compare_by_identity, result)
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

      # Moves every Tensor and buffer this module holds, and every
      # descendant, to another backend, e.g. model.to(:mlx).
      def to(backend)
        target = Backends.fetch(backend)
        each_member { |_, v| v.to(target) }
        self.class.buffer_names.each do |name|
          value = instance_variable_get("@#{name}")
          next if value.nil? || target.native?(value)

          instance_variable_set("@#{name}", target.from(Module.host_copy(value)))
        end
        self
      end

      # The model's state, as a Hash from names to host (Numo) arrays: every
      # parameter, named as in #named_parameters, and every buffer
      # ("layers.1.running_mean"). The arrays are copies, so training on
      # doesn't change them. Save it with Maurograd::SafeTensors.save.
      def state_dict
        dict = {}
        named_parameters.each { |name, t| dict[name] = Module.host_copy(t.data) }
        each_buffer { |name, value| dict[name] = Module.host_copy(value) }
        dict
      end

      # Loads a #state_dict (or a Hash read with Maurograd::SafeTensors.load)
      # into this model's parameters and buffers, on their current backend.
      # Parameters keep their identity, so optimizers holding them keep
      # working.
      #
      # Every key and shape is checked before anything is changed: with
      # strict: true (the default), a missing or unexpected key raises
      # ArgumentError, as does a shape mismatch in either mode. With strict:
      # false, keys present on only one side are skipped.
      def load_state_dict(dict, strict: true)
        dict = dict.transform_keys(&:to_s)
        targets = {}
        named_parameters.each { |name, t| targets[name] = [t.shape, ->(v) { t.data = t.backend.from(v) }] }
        each_buffer do |name, value, owner, ivar|
          backend = Backends.loaded.find { |b| b.native?(value) }
          targets[name] = [value.shape, ->(v) { owner.instance_variable_set(ivar, backend.from(v)) }]
        end

        if strict
          missing = targets.keys - dict.keys
          unexpected = dict.keys - targets.keys
          unless missing.empty? && unexpected.empty?
            raise ArgumentError, "state dict doesn't match the model: missing #{missing.inspect}, unexpected #{unexpected.inspect}"
          end
        end

        updates = targets.filter_map do |name, (shape, assign)|
          next unless dict.key?(name)

          value = ::Numo::SFloat.cast(dict[name]).dup
          unless value.shape == shape
            raise ArgumentError, "#{name}: the state dict has shape #{value.shape.inspect}, the model #{shape.inspect}"
          end

          [assign, value]
        end
        updates.each { |assign, value| assign.call(value) }
        self
      end

      # A host (Numo) copy of a native array of any loaded backend.
      def self.host_copy(value)
        backend = Backends.loaded.find { |b| b.native?(value) }
        raise ArgumentError, "not an array of a loaded backend: #{value.class}" unless backend

        backend.to_numo(value).dup
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

      def collect_modules(name, seen, result)
        return if seen.key?(self)

        seen[self] = true
        result << [name, self]
        named_children.each { |child_name, c| c.collect_modules(name.empty? ? child_name : "#{name}.#{child_name}", seen, result) }
      end

      private

      # Yields [full name, value, module, instance variable] for every
      # buffer of this module and its descendants.
      def each_buffer
        named_modules.each do |prefix, m|
          m.class.buffer_names.each do |name|
            ivar = "@#{name}"
            value = m.instance_variable_get(ivar)
            next if value.nil?

            yield (prefix.empty? ? name : "#{prefix}.#{name}"), value, m, ivar
          end
        end
      end

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
