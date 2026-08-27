require "net/http"
require "uri"
require "fileutils"
require "zlib"
require "digest"
require "numo/narray"

module Maurograd
  module Datasets
    module MNIST
      MIRRORS = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://yann.lecun.com/exdb/mnist/", # Try HTTPS first
        "http://yann.lecun.com/exdb/mnist/"
      ].freeze

      FILES = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
      ].freeze

      MD5 = {
        "train-images-idx3-ubyte.gz" => "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        "train-labels-idx1-ubyte.gz" => "d53e105ee54ea40749a09fcbcd1e9432",
        "t10k-images-idx3-ubyte.gz" => "9fb629c4189551a2d022fa330f9573f3",
        "t10k-labels-idx1-ubyte.gz" => "ec29112dd5afa0611ce80d1b7f02629c"
      }.freeze

      def self.fetch(dest_path:, urls:, expected_md5: nil,
        force: false, timeout: 30)
        # If it already exists and (if requested) the MD5 checks out, return right away.
        if !force && File.exist?(dest_path)
          if expected_md5
            got = Digest::MD5.file(dest_path).hexdigest
            return dest_path if got == expected_md5
            # MD5 mismatch: re-download below.
          else
            return dest_path
          end
        end

        FileUtils.mkdir_p(File.dirname(dest_path))
        tmp = dest_path + ".tmp"
        errors = []

        # Try each mirror/URL.
        Array(urls).each do |u|
          uri = u.is_a?(URI) ? u : URI.parse(u.to_s)
          begin
            File.open(tmp, "wb") do |f|
              Net::HTTP.start(
                uri.host,
                uri.port,
                use_ssl: uri.scheme == "https",
                open_timeout: timeout,
                read_timeout: timeout
              ) do |http|
                req = Net::HTTP::Get.new(uri)
                http.request(req) do |res|
                  unless res.is_a?(Net::HTTPSuccess)
                    raise "HTTP #{res.code} #{res.message} for #{uri}"
                  end
                  res.read_body { |chunk| f.write(chunk) }
                end
              end
            end

            if expected_md5
              got = Digest::MD5.file(tmp).hexdigest
              if got != expected_md5
                raise "MD5 mismatch for #{File.basename(dest_path)}: got #{got}, expected #{expected_md5}"
              end
            end

            FileUtils.mv(tmp, dest_path)
            return dest_path
          rescue => e
            errors << "#{uri} -> #{e.class}: #{e.message}"
            FileUtils.rm_f(tmp)
            next
          end
        end

        raise "All mirrors failed for #{dest_path}:\n" + errors.map { |x| "  - #{x}" }.join("\n")
      end
      private_class_method :fetch

      def self.download(root: "data", force: false, verify_md5: true)
        raw_dir = File.join(root, "mnist", "raw")
        FileUtils.mkdir_p(raw_dir)

        FILES.each do |gz_name|
          gz_path = File.join(raw_dir, gz_name)
          out_path = gz_path.sub(/\.gz\z/, "")

          next if !force && File.exist?(out_path)

          urls = MIRRORS.map { |base| URI.join(base, gz_name).to_s }
          expected = verify_md5 ? MD5[gz_name] : nil

          fetch(
            dest_path: gz_path,
            urls: urls,
            expected_md5: expected,
            force: force
          )

          gunzip(gz_path, out_path)
        end

        raw_dir
      end

      def self.load(root: "data", split: :train, normalize: true)
        raw_dir = File.join(root, "mnist", "raw")

        img_file, lbl_file =
          case split
          when :train
            ["train-images-idx3-ubyte", "train-labels-idx1-ubyte"]
          when :test
            ["t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
          else
            raise ArgumentError, "split must be :train or :test"
          end

        x = read_idx_images(File.join(raw_dir, img_file)) # uint8 -> Numo::UInt8 [N,28,28]
        y = read_idx_labels(File.join(raw_dir, lbl_file)) # uint8 -> Numo::UInt8 [N]

        x = if normalize
          Numo::SFloat.cast(x) / 255.0 # [0,1]
        else
          Numo::SFloat.cast(x)
        end

        # Often useful: add a channel dimension for CNNs: [N, 1, 28, 28]
        x = x.reshape(x.shape[0], 1, 28, 28)

        [x, y] # y can be left as UInt8 (class indices)
      end

      # ------------------------------------------------------------

      def self.download_file(uri, dest_path)
        tmp = dest_path + ".tmp"
        File.open(tmp, "wb") do |f|
          Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https") do |http|
            req = Net::HTTP::Get.new(uri)
            http.request(req) do |res|
              raise "HTTP #{res.code} for #{uri}" unless res.is_a?(Net::HTTPSuccess)
              res.read_body { |chunk| f.write(chunk) }
            end
          end
        end
        FileUtils.mv(tmp, dest_path)
      end
      private_class_method :download_file

      def self.gunzip(gz_path, out_path)
        Zlib::GzipReader.open(gz_path) do |gz|
          File.open(out_path, "wb") { |f| IO.copy_stream(gz, f) }
        end
        out_path
      end
      private_class_method :gunzip

      # IDX format: big-endian integers in header.
      # Images: magic=2051, then n, rows, cols, then uint8 pixels.
      def self.read_idx_images(path)
        buf = File.binread(path)
        magic, n, rows, cols = buf[0, 16].unpack("N4")
        raise "Bad magic #{magic} in #{path}" unless magic == 2051
        data = buf[16, n * rows * cols].bytes
        Numo::UInt8.asarray(data).reshape(n, rows, cols)
      end
      private_class_method :read_idx_images

      # Labels: magic=2049, then n, then uint8 labels.
      def self.read_idx_labels(path)
        buf = File.binread(path)
        magic, n = buf[0, 8].unpack("N2")
        raise "Bad magic #{magic} in #{path}" unless magic == 2049
        data = buf[8, n].bytes
        Numo::UInt8.asarray(data)
      end
      private_class_method :read_idx_labels
    end
  end
end
