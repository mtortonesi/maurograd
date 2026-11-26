require "maurograd"

# Checks the C MaxPool2D kernels against a deliberately naive pure-Ruby
# reference (zero padding, first maximum wins), across shapes that exercise
# overlapping windows, padding and negative inputs.
describe "MaxPool2D C kernels" do
  # Returns [y, dx] for upstream gradient dy, computed with plain loops.
  def reference(x, dy, kh, kw, stride, pad)
    n, c, h, w = x.shape
    oh = (h + 2 * pad - kh) / stride + 1
    ow = (w + 2 * pad - kw) / stride + 1
    y = Numo::SFloat.zeros(n, c, oh, ow)
    dx = Numo::SFloat.zeros(n, c, h, w)

    n.times do |ni|
      c.times do |ci|
        oh.times do |oy|
          ow.times do |ox|
            best = nil
            best_pos = nil
            kh.times do |ky|
              kw.times do |kx|
                iy = oy * stride - pad + ky
                ix = ox * stride - pad + kx
                inside = iy >= 0 && iy < h && ix >= 0 && ix < w
                v = inside ? x[ni, ci, iy, ix] : 0.0
                if best.nil? || v > best
                  best = v
                  best_pos = inside ? [iy, ix] : nil
                end
              end
            end
            y[ni, ci, oy, ox] = best
            dx[ni, ci, *best_pos] += dy[ni, ci, oy, ox] if best_pos
          end
        end
      end
    end

    [y, dx]
  end

  [
    # [shape, kh, kw, stride, pad]
    [[2, 3, 8, 8], 2, 2, 2, 0],   # the LeNet case
    [[2, 2, 7, 9], 3, 3, 2, 0],   # overlapping windows, odd sizes
    [[1, 2, 5, 5], 3, 2, 1, 1],   # rectangular kernel, stride 1, padding
    [[3, 1, 6, 6], 2, 2, 2, 1]    # padding with non-overlapping windows
  ].each do |shape, kh, kw, stride, pad|
    it "matches the reference for shape #{shape}, kernel #{kh}x#{kw}, stride #{stride}, pad #{pad}" do
      Maurograd.seed!(3)
      xd = Numo::SFloat.new(*shape).rand_norm # includes negatives, so padding can win
      x = Maurograd::Tensor.new(xd, requires_grad: true)

      y = Maurograd::Ops::MaxPool2D.apply(x, kh, kw, stride, pad)
      dy = Numo::SFloat.new(*y.shape).rand_norm
      y.backward(dy)

      ref_y, ref_dx = reference(xd, dy, kh, kw, stride, pad)
      expect(y.data).to be == ref_y
      expect((x.grad - ref_dx).abs.max).to be < 1e-5
    end
  end

  it "sends a tied window's gradient to the first maximum only" do
    x = Maurograd::Tensor.new(Numo::SFloat.ones(1, 1, 2, 2), requires_grad: true)
    y = Maurograd::Ops::MaxPool2D.apply(x, 2, 2, 2, 0)
    y.backward(Numo::SFloat[[[[1.0]]]])

    expect(x.grad).to be == Numo::SFloat[[[[1, 0], [0, 0]]]]
  end

  it "sends no gradient anywhere when a padding cell is the maximum" do
    x = Maurograd::Tensor.new(Numo::SFloat.new(1, 1, 2, 2).fill(-1.0), requires_grad: true)
    y = Maurograd::Ops::MaxPool2D.apply(x, 2, 2, 2, 1)
    expect(y.data).to be == Numo::SFloat.zeros(1, 1, 2, 2)

    y.backward(Numo::SFloat.ones(1, 1, 2, 2))
    expect(x.grad).to be == Numo::SFloat.zeros(1, 1, 2, 2)
  end

  it "handles non-contiguous inputs" do
    base = Numo::SFloat.new(2, 8, 8, 3).rand_norm
    xd = base.transpose(0, 3, 1, 2) # NHWC -> NCHW view, not contiguous
    expect(xd.contiguous?).to be == false

    x = Maurograd::Tensor.new(xd, requires_grad: true)
    y = Maurograd::Ops::MaxPool2D.apply(x, 2, 2, 2, 0)
    ref_y, = reference(xd.copy, Numo::SFloat.zeros(*y.shape), 2, 2, 2, 0)
    expect(y.data).to be == ref_y
  end
end
