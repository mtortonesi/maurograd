import os, time
import torch

def now():
    return time.perf_counter()

torch.manual_seed(0)
torch.set_num_threads(1)

n    = int(os.environ.get("N", 32))
cin  = int(os.environ.get("CIN", 1))
h    = int(os.environ.get("H", 28))
w    = int(os.environ.get("W", 28))
cout = int(os.environ.get("COUT", 8))
kh   = int(os.environ.get("KH", 3))
kw   = int(os.environ.get("KW", 3))
stride = int(os.environ.get("STRIDE", 1))
pad    = int(os.environ.get("PAD", 1))

iters  = int(os.environ.get("ITERS", 200))
warmup = int(os.environ.get("WARMUP", 50))

x = torch.randn((n, cin, h, w), dtype=torch.float32, requires_grad=True)
conv = torch.nn.Conv2d(cin, cout, (kh, kw), stride=stride, padding=pad, bias=True)

# Warmup
for _ in range(warmup):
    y = conv(x)
    loss = y.sum()
    loss.backward()
    conv.zero_grad(set_to_none=True)
    x.grad = None

t_fwd = 0.0
t_bwd = 0.0

for _ in range(iters):
    conv.zero_grad(set_to_none=True)
    x.grad = None

    t0 = now()
    y = conv(x)
    t1 = now()
    loss = y.sum()
    loss.backward()
    t2 = now()

    t_fwd += (t1 - t0)
    t_bwd += (t2 - t1)

print({
    "impl": "pytorch",
    "n": n, "cin": cin, "cout": cout, "h": h, "w": w, "kh": kh, "kw": kw, "stride": stride, "pad": pad,
    "iters": iters,
    "fwd_ms_avg": (t_fwd / iters) * 1000.0,
    "bwd_ms_avg": (t_bwd / iters) * 1000.0,
    "total_ms_avg": ((t_fwd + t_bwd) / iters) * 1000.0
})

