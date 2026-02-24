require 'tmpdir'
require 'fileutils'
require 'numo/narray'

require 'maurograd/datasets/mnist'

describe Maurograd::Datasets::MNIST do
  def idx_images_bytes(n, rows, cols)
    header = [2051, n, rows, cols].pack("N4")
    payload = Array.new(n * rows * cols) { |i| i % 256 }.pack("C*")
    header + payload
  end

  def idx_labels_bytes(n)
    header = [2049, n].pack("N2")
    payload = Array.new(n) { |i| i % 10 }.pack("C*")
    header + payload
  end

  def write_raw(path, raw_bytes)
    File.binwrite(path, raw_bytes)
  end

  it "loads train set from local IDX files, returning images [N,1,28,28] and labels [N]" do
    Dir.mktmpdir do |dir|
      raw_dir = File.join(dir, "mnist", "raw")
      FileUtils.mkdir_p(raw_dir)

      n = 3
      rows = 28
      cols = 28

      images = File.join(raw_dir, "train-images-idx3-ubyte")
      labels = File.join(raw_dir, "train-labels-idx1-ubyte")

      write_raw(images, idx_images_bytes(n, rows, cols))
      write_raw(labels, idx_labels_bytes(n))

      # sanity: se fallisce qui, il problema è nei path
      expect(File.exist?(images)).to be == true
      expect(File.exist?(labels)).to be == true

      x, y = Maurograd::Datasets::MNIST.load(root: dir, split: :train, normalize: true)

      expect(x.shape).to be == [n, 1, 28, 28]
      expect(y.shape).to be == [n]
      expect(x.min).to be >= 0.0
      expect(x.max).to be <= 1.0
    end
  end

  it "supports normalize: false (raw 0..255 scale)" do
    Dir.mktmpdir do |dir|
      raw_dir = File.join(dir, "mnist", "raw")
      FileUtils.mkdir_p(raw_dir)

      n = 2
      rows = 28
      cols = 28

      images = File.join(raw_dir, "t10k-images-idx3-ubyte")
      labels = File.join(raw_dir, "t10k-labels-idx1-ubyte")

      write_raw(images, idx_images_bytes(n, rows, cols))
      write_raw(labels, idx_labels_bytes(n))

      expect(File.exist?(images)).to be == true
      expect(File.exist?(labels)).to be == true

      x, _y = Maurograd::Datasets::MNIST.load(root: dir, split: :test, normalize: false)

      expect(x.shape).to be == [n, 1, 28, 28]
      expect(x.min).to be >= 0
      expect(x.max).to be <= 255
    end
  end
end

