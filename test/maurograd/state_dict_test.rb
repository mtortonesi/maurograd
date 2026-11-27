require "maurograd"
require "tmpdir"

describe "Module#state_dict and #load_state_dict" do
  def build(seed)
    Maurograd.seed!(seed)
    Maurograd::Layers::Sequential.new(
      Maurograd::Layers::Conv2D.new(1, 2, 3, padding: 1), Maurograd::Layers::BatchNorm2D.new(2),
      Maurograd::Layers::ReLU.new, Maurograd::Layers::Flatten.new, Maurograd::Layers::Linear.new(32, 3)
    )
  end

  def input
    Maurograd.seed!(99)
    Maurograd::Tensor.new(Numo::SFloat.new(4, 1, 4, 4).rand_norm)
  end

  it "holds every parameter and buffer, by name, as host arrays" do
    dict = build(1).state_dict
    expect(dict.keys).to be == %w[
      layers.0.weights layers.0.bias layers.1.gamma layers.1.beta layers.4.weights layers.4.bias
      layers.1.running_mean layers.1.running_var
    ]
    expect(dict.values.all? { |v| v.is_a?(Numo::SFloat) }).to be == true
    expect(dict["layers.0.weights"].shape).to be == [2, 1, 3, 3]
  end

  it "copies the arrays, so training on doesn't change a saved state" do
    model = build(1)
    dict = model.state_dict
    before = dict["layers.4.weights"].dup
    model.parameters.each { |p| p.data = p.data + 1 }
    expect(dict["layers.4.weights"]).to be == before
  end

  it "restores a model exactly, buffers included, keeping parameter identity" do
    trained = build(1)
    trained.call(input) # updates BatchNorm's running statistics
    fresh = build(2)
    weights = fresh.parameters

    fresh.load_state_dict(trained.state_dict)

    expect(fresh.parameters.map(&:object_id)).to be == weights.map(&:object_id)
    trained.state_dict.each { |name, value| expect(fresh.state_dict[name]).to be == value }
    expect(fresh.eval.call(input).data).to be == trained.eval.call(input).data
  end

  it "checks keys and shapes before changing anything" do
    model = build(1)
    before = model.state_dict

    missing = before.except("layers.4.bias")
    expect { model.load_state_dict(missing) }.to raise_exception(ArgumentError, message: be =~ /missing \["layers.4.bias"\]/)
    expect { model.load_state_dict(before.merge("extra" => Numo::SFloat[1])) }.to raise_exception(ArgumentError, message: be =~ /unexpected \["extra"\]/)

    wrong = build(2).state_dict.merge("layers.4.bias" => Numo::SFloat.zeros(4))
    expect { model.load_state_dict(wrong) }.to raise_exception(ArgumentError, message: be =~ /layers.4.bias/)
    model.state_dict.each { |name, value| expect(value).to be == before[name] }
  end

  it "loads a partial state with strict: false" do
    model = build(1)
    model.load_state_dict({"layers.4.bias" => Numo::SFloat[7, 8, 9]}, strict: false)
    expect(model.state_dict["layers.4.bias"].to_a).to be == [7.0, 8.0, 9.0]
  end

  it "inherits buffers declared by a superclass" do
    expect(Class.new(Maurograd::Layers::BatchNorm2D).buffer_names).to be == %w[running_mean running_var]
    expect(Maurograd::Layers::Linear.buffer_names).to be == []
  end
end

describe Maurograd::SafeTensors do
  def tmp(name)
    File.join(@dir, name)
  end

  def around(&block)
    Dir.mktmpdir do |dir|
      @dir = dir
      super
    end
  end

  it "round-trips arrays of several dtypes and shapes" do
    tensors = {
      "w" => Numo::SFloat.new(2, 3).rand_norm,
      "d" => Numo::DFloat[1.5, -2.25],
      "i" => Numo::Int32[[1, -2], [3, 4]],
      "b" => Numo::UInt8[0, 255],
      "s" => Numo::SFloat.new(0)
    }
    Maurograd::SafeTensors.save(tensors, tmp("t.safetensors"))
    loaded = Maurograd::SafeTensors.load(tmp("t.safetensors"))

    expect(loaded.keys).to be == tensors.keys
    tensors.each do |name, value|
      expect(loaded[name].class).to be == value.class
      expect(loaded[name].shape).to be == value.shape
      expect(loaded[name]).to be == value
    end
  end

  it "writes the layout of the format: header size, JSON header, raw bytes" do
    Maurograd::SafeTensors.save({"x" => Numo::SFloat[1, 2]}, tmp("x.safetensors"), metadata: {format: "pt"})
    bytes = File.binread(tmp("x.safetensors"))
    n = bytes[0, 8].unpack1("Q<")

    expect(n % 8).to be == 0
    header = JSON.parse(bytes[8, n])
    expect(header).to be == {"x" => {"dtype" => "F32", "shape" => [2], "data_offsets" => [0, 8]}, "__metadata__" => {"format" => "pt"}}
    expect(bytes[8 + n..].unpack("e*")).to be == [1.0, 2.0]
  end

  it "saves and loads a model's state" do
    Maurograd.seed!(1)
    model = Maurograd::Layers::Sequential.new(Maurograd::Layers::Linear.new(3, 2))
    Maurograd::SafeTensors.save(model.state_dict, tmp("m.safetensors"))

    Maurograd.seed!(2)
    other = Maurograd::Layers::Sequential.new(Maurograd::Layers::Linear.new(3, 2))
    other.load_state_dict(Maurograd::SafeTensors.load(tmp("m.safetensors")))
    expect(other.state_dict).to be == model.state_dict
  end

  it "rejects malformed files instead of reading past them" do
    path = tmp("bad.safetensors")
    write = lambda do |header, data = ""|
      json = JSON.generate(header)
      File.binwrite(path, [json.bytesize].pack("Q<") + json + data)
    end

    File.binwrite(path, "abc")
    expect { Maurograd::SafeTensors.load(path) }.to raise_exception(ArgumentError)

    File.binwrite(path, [1 << 40].pack("Q<") + "{}")
    expect { Maurograd::SafeTensors.load(path) }.to raise_exception(ArgumentError)

    write.call({"x" => {"dtype" => "F32", "shape" => [4], "data_offsets" => [0, 16]}}, "\0" * 8)
    expect { Maurograd::SafeTensors.load(path) }.to raise_exception(ArgumentError, message: be =~ /data offsets/)

    write.call({"x" => {"dtype" => "BF16", "shape" => [1], "data_offsets" => [0, 2]}}, "\0\0")
    expect { Maurograd::SafeTensors.load(path) }.to raise_exception(ArgumentError, message: be =~ /BF16/)

    File.binwrite(path, [2].pack("Q<") + "{x")
    expect { Maurograd::SafeTensors.load(path) }.to raise_exception(ArgumentError, message: be =~ /JSON/)
  end
end
