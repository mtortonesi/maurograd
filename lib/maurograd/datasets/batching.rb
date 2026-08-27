module Maurograd
  module Datasets
    module Batching
      def self.batches(x, y, batch_size: 64, shuffle: true, seed: 0, drop_last: false)
        n = x.shape[0]
        idx = (0...n).to_a

        if shuffle
          rng = Random.new(seed)
          idx = idx.sort_by { rng.rand }
        end

        Enumerator.new do |enum|
          idx.each_slice(batch_size) do |ii|
            next if drop_last && ii.length < batch_size

            xb = x[ii, true, true, true]
            yb = y[ii]
            enum << [xb, yb]
          end
        end
      end
    end
  end
end
