#!/usr/bin/env ruby
# bench/bench_torchrb_conv.rb
require "torch"

def now
  Process.clock_gettime(Process::CLOCK_MONOTONIC)
end

Torch.manual_seed(0)
Torch.set_num_threads(1) if Torch.respond_to?(:set_num_threads)

n = Integer(ENV.fetch("N", "32"))
cin = Integer(ENV.fetch("CIN", "1"))
h = Integer(ENV.fetch("H", "28"))
w = Integer(ENV.fetch("W", "28"))
cout = Integer(ENV.fetch("COUT", "8"))
kh = Integer(ENV.fetch("KH", "3"))
kw = Integer(ENV.fetch("KW", "3"))
stride = Integer(ENV.fetch("STRIDE", "1"))
padding = Integer(ENV.fetch("PAD", "1"))

iters = Integer(ENV.fetch("ITERS", "50"))
warmup = Integer(ENV.fetch("WARMUP", "10"))

x = Torch.randn([n, cin, h, w], dtype: :float32).requires_grad!
conv = Torch::NN::Conv2d.new(cin, cout, [kh, kw], stride: stride, padding: padding, bias: true)
conv.train

# Warmup
warmup.times do
  y = conv.call(x)
  loss = y.sum
  loss.backward
  conv.zero_grad
  x.grad&.zero!
end

t_fwd = 0.0
t_bwd = 0.0

iters.times do
  conv.zero_grad
  x.grad&.zero!

  t0 = now
  y = conv.call(x)
  t1 = now

  loss = y.sum
  loss.backward
  t2 = now

  t_fwd += (t1 - t0)
  t_bwd += (t2 - t1)
end

puts({
  impl: "torchrb_libtorch",
  n: n, cin: cin, cout: cout, h: h, w: w, kh: kh, kw: kw, stride: stride, pad: padding,
  iters: iters,
  fwd_ms_avg: (t_fwd / iters) * 1000.0,
  bwd_ms_avg: (t_bwd / iters) * 1000.0,
  total_ms_avg: ((t_fwd + t_bwd) / iters) * 1000.0
}.inspect)
