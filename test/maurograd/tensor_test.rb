describe Maurograd::Tensor do
  let(:data) { Numo::SFloat.new(2, 2).seq }
  let(:tensor) { Maurograd::Tensor.new(data) }

  it 'has the correct shape' do
    expect(tensor.shape).to be == [2, 2]
  end

  # it 'supports requires_grad' do
  #   t = Maurograd::Tensor.new(data, requires_grad: true)
  #   expect(t.grad).not.to be_nil
  # end

  # describe "Multidimensional tensor transpose" do
  #   it "checks the behavior of transpose on 3D tensors" do
  #     # Create a [2, 3, 4] tensor -> Batch=2, Rows=3, Cols=4
  #     data = Numo::SFloat.new(2, 3, 4).seq

  #     # Numo's standard transpose reverses everything -> [4, 3, 2]
  #     expect(data.transpose.shape).to be == [4, 3, 2]

  #     # For MatMul we only want to swap the last two -> [2, 4, 3]
  #     # In Numo this is done by passing the axis indices: [0, 2, 1]
  #     batch_transpose = data.transpose(0, 2, 1)
  #     expect(batch_transpose.shape).to be == [2, 4, 3]
  #   end
  # end

  # We want to check that transposing a multidimensional tensor only
  # swaps the last two dimensions (H and W), leaving the others (N and C) intact.
  describe "4D tensor manipulation" do
    it "swaps only the spatial dimensions H and W" do
      # Shape: [N=2, C=3, H=4, W=5]
      data = Numo::SFloat.new(2, 3, 4, 5).seq

      # We want to go from [0, 1, 2, 3] to [0, 1, 3, 2]
      # This swaps H and W while leaving N and C intact.
      res = data.transpose(0, 1, 3, 2)

      expect(res.shape).to be == [2, 3, 5, 4]
    end
  end

  # Simple addition test (no broadcasting).
  describe 'simple addition operation' do
    let(:a) { Maurograd::Tensor.new(Numo::SFloat[1, 2, 3], requires_grad: true) }
    let(:b) { Maurograd::Tensor.new(Numo::SFloat[4, 5, 6], requires_grad: true) }
    let(:c) { a + b }

    it 'correctly computes the forward pass' do
      expect(c.data).to be == Numo::SFloat[5, 7, 9]
    end

    it 'correctly accumulates gradients when a tensor is used more than once' do
      a = Maurograd::Tensor.new(Numo::SFloat[2, 3], requires_grad: true)
      # Operation: b = a + a
      b = a + a

      b.backward(Numo::SFloat[1, 1])

      # The derivative of x + x is 2, so the gradient must be [2, 2].
      expect(a.grad).to be == Numo::SFloat[2, 2]
    end

    it 'correctly computes the gradients in the backward pass' do
      c.backward(Numo::SFloat[1, 1, 1])

      expect(a.grad).to be == Numo::SFloat[1, 1, 1]
      expect(b.grad).to be == Numo::SFloat[1, 1, 1]
    end
  end

  # Broadcasting test (essential for CNN biases).
  describe 'gradient broadcasting' do
    # Imagine a mini-batch of 2 samples, with 3 channels (e.g. 2x3).
    let(:data) { Numo::SFloat[[1, 2, 3], [4, 5, 6]] }
    let(:bias_data) { Numo::SFloat[[10, 20, 30]] } # Shape [1, 3]

    let(:a) { Maurograd::Tensor.new(data, requires_grad: true) }
    let(:b) { Maurograd::Tensor.new(bias_data, requires_grad: true) }
    let(:c) { a + b }

    it 'correctly broadcasts in the forward pass' do
      expected = Numo::SFloat[[11, 22, 33], [14, 25, 36]]
      expect(c.data).to be == expected
    end

    it 'correctly reduces the gradient for the broadcasted tensor' do
      # Output gradient (same shape as c: 2x3).
      grad_output = Numo::SFloat[[1, 1, 1], [1, 1, 1]]
      c.backward(grad_output)

      # For 'a', the gradient must be identical to grad_output (2x3).
      expect(a.grad).to be == grad_output

      # For 'b' (the bias), the gradient must be the sum along the batch axis.
      # Expected result: [1+1, 1+1, 1+1] -> [2, 2, 2] with shape [1, 3].
      expect(b.grad.shape).to be == [1, 3]
      expect(b.grad).to be == Numo::SFloat[[2, 2, 2]]
    end
  end
end
