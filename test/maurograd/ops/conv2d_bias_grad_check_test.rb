# frozen_string_literal: true

require 'maurograd/tensor'
require 'maurograd/layers/conv2d'


describe Maurograd::Ops::Conv2D do
  it "matches numerical gradient for one bias element (tiny conv)" do
    srand 1234

    # Tiny shapes: N=2, C=3, H=W=5, Cout=4, Kh=Kw=3
    n  = 2
    c  = 3
    h  = 5
    w  = 5
    cout = 4
    kh = 3
    kw = 3
    stride = 1
    pad = 1

    # Random but small to keep numerics stable
    x = Numo::SFloat.new(n, c, h, w).rand - 0.5
    ww = (Numo::SFloat.new(cout, c, kh, kw).rand - 0.5) * 0.1
    b = (Numo::SFloat.new(cout).rand - 0.5) * 0.1

    xt = Maurograd::Tensor.new(x, requires_grad: false)
    wt = Maurograd::Tensor.new(ww, requires_grad: false)
    bt = Maurograd::Tensor.new(b, requires_grad: true)

    # Loss: sum of all outputs => dL/db[co] should be N*OH*OW for that co (since dout is ones)
    out = Maurograd::Ops::Conv2D.apply(xt, wt, bt, stride, pad)
    loss = out.sum # assume you have Tensor#sum returning scalar Tensor
    loss.backward

    raise "bias.grad is nil" if bt.grad.nil?

    co = 1          # check bias element 1 (arbitrary)
    eps = 1e-3

    # Numerical gradient
    b_plus = b.dup
    b_minus = b.dup
    b_plus[co]  += eps
    b_minus[co] -= eps

    out_plus = Maurograd::Ops::Conv2D.apply(
      Maurograd::Tensor.new(x, requires_grad: false),
      Maurograd::Tensor.new(ww, requires_grad: false),
      Maurograd::Tensor.new(b_plus, requires_grad: false),
      stride, pad
    )
    loss_plus = out_plus.sum.data.to_f

    out_minus = Maurograd::Ops::Conv2D.apply(
      Maurograd::Tensor.new(x, requires_grad: false),
      Maurograd::Tensor.new(ww, requires_grad: false),
      Maurograd::Tensor.new(b_minus, requires_grad: false),
      stride, pad
    )
    loss_minus = out_minus.sum.data.to_f

    g_num = (loss_plus - loss_minus) / (2.0 * eps)
    g_aut = bt.grad[co].to_f

    # Tolerance: for float32 + finite differences, 1e-2 is reasonable; tighten to 1e-3 if everything is very stable
    err = (g_aut - g_num).abs
    assert err < 1e-2, "bias grad mismatch: aut=#{g_aut} num=#{g_num} err=#{err}"
  end
end

