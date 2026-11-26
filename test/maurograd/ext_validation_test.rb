require "maurograd"

# The C extension reads raw float32 memory, so its public entry points must
# reject anything that would make it read the wrong memory or the wrong
# amount of it, instead of returning garbage (or crashing).
describe "Maurograd::Ext argument validation" do
  let(:x) { Numo::SFloat.new(1, 1, 8, 8).seq }

  it "col2im rejects a dcol whose shape doesn't match x_shape" do
    # The originally reported case: a 1x1 dcol for an 8x8 input used to be
    # read past its end. im2col(x, 3, 3, 1, 0) is [36, 9].
    expect { Maurograd::Ext.col2im(Numo::SFloat.zeros(1, 1), [1, 1, 8, 8], 3, 3, 1, 0) }
      .to raise_exception(ArgumentError, message: be =~ /dcol must have shape \[36, 9\]/)
    expect { Maurograd::Ext.col2im(Numo::SFloat.zeros(36, 8), [1, 1, 8, 8], 3, 3, 1, 0) }
      .to raise_exception(ArgumentError)
    expect { Maurograd::Ext.col2im(Numo::SFloat.zeros(36 * 9), [1, 1, 8, 8], 3, 3, 1, 0) }
      .to raise_exception(ArgumentError)
  end

  it "col2im rejects a malformed x_shape" do
    dcol = Numo::SFloat.zeros(36, 9)
    expect { Maurograd::Ext.col2im(dcol, [1, 8, 8], 3, 3, 1, 0) }.to raise_exception(ArgumentError)
    expect { Maurograd::Ext.col2im(dcol, [1, 0, 8, 8], 3, 3, 1, 0) }.to raise_exception(ArgumentError)
    expect { Maurograd::Ext.col2im(dcol, "1x1x8x8", 3, 3, 1, 0) }.to raise_exception(TypeError)
  end

  it "col2im and im2col still round-trip valid input" do
    col = Maurograd::Ext.im2col(x, 3, 3, 1, 0)
    expect(col.shape).to be == [36, 9]
    expect(Maurograd::Ext.col2im(col, [1, 1, 8, 8], 3, 3, 1, 0).shape).to be == [1, 1, 8, 8]
  end

  it "rejects non-contiguous arrays instead of reading them as contiguous" do
    strided = Numo::SFloat.new(1, 1, 8, 16).seq[true, true, true, (0..15).step(2)]
    expect(strided.contiguous?).to be == false
    expect { Maurograd::Ext.im2col(strided, 3, 3, 1, 0) }
      .to raise_exception(ArgumentError, message: be =~ /contiguous/)

    dcol = Numo::SFloat.zeros(9, 36).transpose # [36, 9], but a transposed view
    expect { Maurograd::Ext.col2im(dcol, [1, 1, 8, 8], 3, 3, 1, 0) }
      .to raise_exception(ArgumentError, message: be =~ /contiguous/)

    w = Numo::SFloat.ones(2, 1, 3, 3)
    expect { Maurograd::Ext.conv2d_forward(strided, w, nil, 1, 0, {}) }
      .to raise_exception(ArgumentError, message: be =~ /contiguous/)
  end

  it "rejects zero stride, non-positive kernels and negative padding" do
    expect { Maurograd::Ext.im2col(x, 3, 3, 0, 0) }.to raise_exception(ArgumentError, message: be =~ /stride/)
    expect { Maurograd::Ext.im2col(x, 0, 3, 1, 0) }.to raise_exception(ArgumentError, message: be =~ /kernel/)
    expect { Maurograd::Ext.im2col(x, 3, 3, 1, -1) }.to raise_exception(ArgumentError, message: be =~ /padding/)
    expect { Maurograd::Ext.col2im(Numo::SFloat.zeros(36, 9), [1, 1, 8, 8], 3, 3, 0, 0) }
      .to raise_exception(ArgumentError, message: be =~ /stride/)
  end

  it "makes Conv2D with stride 0 raise instead of returning garbage" do
    conv = Maurograd::Layers::Conv2D.new(1, 2, 3, stride: 0)
    expect { conv.forward(Maurograd::Tensor.new(x)) }.to raise_exception(ArgumentError, message: be =~ /stride/)
  end

  it "still accepts strided inputs through Ops::Conv2D, which copies them first" do
    Maurograd.seed!(1)
    base = Numo::SFloat.new(2, 8, 8, 3).rand_norm
    strided = base.transpose(0, 3, 1, 2) # NHWC -> NCHW view
    conv = Maurograd::Layers::Conv2D.new(3, 4, 3, padding: 1)

    y_view = conv.forward(Maurograd::Tensor.new(strided)).data
    y_copy = conv.forward(Maurograd::Tensor.new(strided.copy)).data
    expect(y_view).to be == y_copy
  end
end
