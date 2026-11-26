require "maurograd/tensor"
require "maurograd/layers/batchnorm2d"

describe "BatchNorm2D running statistics" do
  it "updates running_mean/running_var via an EMA, using unbiased variance" do
    momentum = 0.1
    bn = Maurograd::Layers::BatchNorm2D.new(2, momentum: momentum)
    # Defaults: running_mean = 0, running_var = 1.

    x = Numo::SFloat.new(2, 2, 3, 3).rand
    input = Maurograd::Tensor.new(x, requires_grad: false)

    n = x.shape[0] * x.shape[2] * x.shape[3]

    bn.forward(input)

    [0, 1].each do |c|
      xc = x[true, c, true, true]
      batch_mean = xc.mean
      biased_var = ((xc - batch_mean)**2).mean
      unbiased_var = biased_var * n.to_f / (n - 1)

      expected_running_mean = (1 - momentum) * 0.0 + momentum * batch_mean
      expected_running_var = (1 - momentum) * 1.0 + momentum * unbiased_var

      expect((bn.running_mean[c] - expected_running_mean).abs).to be < 1e-4
      expect((bn.running_var[c] - expected_running_var).abs).to be < 1e-4
    end
  end
end
