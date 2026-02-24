require 'set'

module Maurograd
  module Scalar
    module Backpropagation
      def zero_grad!
        topological_sort.each { |v| v.gradient = 0.0 }
      end

      def backpropagate
        self.gradient = 1.0
        topo = topological_sort
        topo.reverse_each do |value|
          value.backward&.call
        end
      end

      # DFS with color: temp = gray (in pending stack), seen = black (done)
      def topological_sort(seen = Set.new, temp = Set.new, out = [])
        return out if seen.include?(self)

        if temp.include?(self)
          raise ArgumentError, "Cycle detected at #{self.inspect}"
        end

        temp.add(self)

        (children || []).uniq.each do |child|
          next if child.nil?
          child.topological_sort(seen, temp, out)
        end

        temp.delete(self)
        seen.add(self)
        out << self
        out
      end
    end
  end
end
