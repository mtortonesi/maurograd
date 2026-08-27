#!/usr/bin/env ruby
# bench/bench_maurograd_conv.rb
require "numo/narray"
require_relative "../lib/maurograd/tensor"
require_relative "../lib/maurograd/ops/conv2d"

include Maurograd

def now
  Process.clock_gettime(Process::CLOCK_MONOTONIC)
end

# Params "fair"
n = Integer(ENV.fetch("N", "32"))
cin = Integer(ENV.fetch("CIN", "1"))
h = Integer(ENV.fetch("H", "28"))
w = Integer(ENV.fetch("W", "28"))
cout = Integer(ENV.fetch("COUT", "8"))
kh = Integer(ENV.fetch("KH", "3"))
kw = Integer(ENV.fetch("KW", "3"))
stride = Integer(ENV.fetch("STRIDE", "1"))
padding = Integer(ENV.fetch("PAD", "1"))

iters = Integer(ENV.fetch("ITERS", "20"))
warmup = Integer(ENV.fetch("WARMUP", "5"))

x = Tensor.new(Numo::SFloat.new(n, cin, h, w).rand_norm, requires_grad: true)
wgt = Tensor.new(Numo::SFloat.new(cout, cin, kh, kw).rand_norm, requires_grad: true)
bias = Tensor.new(Numo::SFloat.new(cout).rand_norm, requires_grad: true)

# small scalar loss
def loss_from(y)
  # sum -> scalar
  Tensor.new(y.data.sum, requires_grad: true).tap do |t|
    t.creator = nil # already scalar: simplifying here. Use your Loss op if you have one.
  end
end

# Warmup
warmup.times do
  y = Ops::Conv2D.apply(x, wgt, bias, stride, padding)
  # fake grad: all ones
  dy = Numo::SFloat.ones(*y.shape)
  y.backward(dy)
end

t_fwd = 0.0
t_bwd = 0.0

iters.times do
  # reset grad (in case your Tensor accumulates)
  x.grad = nil if x.respond_to?(:grad=)
  wgt.grad = nil if wgt.respond_to?(:grad=)
  bias.grad = nil if bias.respond_to?(:grad=)

  t0 = now
  y = Ops::Conv2D.apply(x, wgt, bias, stride, padding)
  t1 = now

  dy = Numo::SFloat.ones(*y.shape)
  y.backward(dy)
  t2 = now

  t_fwd += (t1 - t0)
  t_bwd += (t2 - t1)
end

puts({
  impl: "maurograd_numo",
  n: n, cin: cin, cout: cout, h: h, w: w, kh: kh, kw: kw, stride: stride, pad: padding,
  iters: iters,
  fwd_ms_avg: (t_fwd / iters) * 1000.0,
  bwd_ms_avg: (t_bwd / iters) * 1000.0,
  total_ms_avg: ((t_fwd + t_bwd) / iters) * 1000.0
}.inspect)
