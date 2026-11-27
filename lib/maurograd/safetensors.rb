require "json"
require "numo/narray"

module Maurograd
  # Reads and writes the safetensors format, the common file format for
  # model weights (https://github.com/huggingface/safetensors): files saved
  # here can be loaded by Python's safetensors package, and the other way
  # round.
  #
  #   Maurograd::SafeTensors.save(model.state_dict, "lenet.safetensors")
  #   model.load_state_dict(Maurograd::SafeTensors.load("lenet.safetensors"))
  #
  # The format is deliberately simple, and can't run code when loaded
  # (unlike Ruby's Marshal or Python's pickle), so loading a file from
  # someone else is safe:
  #
  #   8 bytes      N, the header size (unsigned 64-bit, little-endian)
  #   N bytes      a JSON header: for each tensor, its dtype, shape and the
  #                byte range of its data, e.g.
  #                {"weights": {"dtype": "F32", "shape": [2, 3],
  #                             "data_offsets": [0, 24]}, ...}
  #   the rest     the tensors' raw bytes, row-major, little-endian
  module SafeTensors
    DTYPES = {
      ::Numo::SFloat => "F32", ::Numo::DFloat => "F64",
      ::Numo::Int8 => "I8", ::Numo::Int16 => "I16", ::Numo::Int32 => "I32", ::Numo::Int64 => "I64",
      ::Numo::UInt8 => "U8", ::Numo::UInt16 => "U16", ::Numo::UInt32 => "U32", ::Numo::UInt64 => "U64"
    }.freeze

    # Larger headers are refused when loading, so that a corrupt or hostile
    # file can't make us allocate gigabytes for one.
    MAX_HEADER_BYTES = 100 * 1024 * 1024

    class << self
      # Writes `tensors`, a Hash from names to Numo arrays (or Tensors, or
      # anything Numo can cast), to `path`. `metadata` is an optional Hash
      # of strings, stored in the header.
      def save(tensors, path, metadata: nil)
        check_little_endian!
        header = {}
        blobs = []
        offset = 0
        tensors.each do |name, value|
          array = host_array(value)
          dtype = DTYPES[array.class] or raise ArgumentError, "#{name}: #{array.class} has no safetensors dtype"
          bytes = array.to_binary
          header[name.to_s] = {"dtype" => dtype, "shape" => array.shape, "data_offsets" => [offset, offset + bytes.bytesize]}
          blobs << bytes
          offset += bytes.bytesize
        end
        header["__metadata__"] = metadata.to_h { |k, v| [k.to_s, v.to_s] } if metadata

        json = JSON.generate(header)
        json += " " * (-json.bytesize % 8) # pad so the data starts 8-byte aligned
        File.open(path, "wb") do |f|
          f.write([json.bytesize].pack("Q<"), json)
          blobs.each { |b| f.write(b) }
        end
        path
      end

      # Reads a safetensors file into a Hash from names to Numo arrays.
      # Raises ArgumentError if the file is malformed.
      def load(path)
        check_little_endian!
        File.open(path, "rb") do |f|
          size_bytes = f.read(8)
          raise ArgumentError, "#{path}: too short to be a safetensors file" unless size_bytes&.bytesize == 8

          n = size_bytes.unpack1("Q<")
          raise ArgumentError, "#{path}: header size #{n} is not plausible" if n > MAX_HEADER_BYTES || n > f.size - 8

          header = JSON.parse(f.read(n))
          raise ArgumentError, "#{path}: the header is not a JSON object" unless header.is_a?(Hash)

          header.delete("__metadata__")
          data = f.read || ""
          header.to_h { |name, info| [name, decode(path, name, info, data)] }
        end
      rescue JSON::ParserError => e
        raise ArgumentError, "#{path}: the header is not valid JSON (#{e.message})"
      end

      private

      def decode(path, name, info, data)
        klass = DTYPES.key(info["dtype"]) or
          raise ArgumentError, "#{path}: #{name} has dtype #{info["dtype"].inspect}, which Numo can't represent"
        shape = info["shape"]
        unless shape.is_a?(Array) && shape.all? { |d| d.is_a?(Integer) && d >= 0 }
          raise ArgumentError, "#{path}: #{name} has an invalid shape #{shape.inspect}"
        end

        from, to = info["data_offsets"]
        expected = shape.reduce(1, :*) * klass::ELEMENT_BYTE_SIZE
        unless from.is_a?(Integer) && to.is_a?(Integer) && from >= 0 && to <= data.bytesize && to - from == expected
          raise ArgumentError, "#{path}: #{name}'s data offsets #{info["data_offsets"].inspect} don't fit its shape or the file"
        end

        klass.from_binary(data.byteslice(from, expected), shape)
      end

      def host_array(value)
        value = value.to_numo if value.is_a?(Tensor)
        value.is_a?(::Numo::NArray) ? value : ::Numo::SFloat.cast(value)
      end

      # Numo reads and writes memory in the machine's byte order, and the
      # format is little-endian, as are x86 and ARM machines.
      def check_little_endian!
        raise NotImplementedError, "safetensors I/O needs a little-endian machine" unless [1].pack("L") == [1].pack("L<")
      end
    end
  end
end
