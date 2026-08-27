describe 'MatMul operation' do
  # Test 1: standard multiplication between 2D matrices.
  describe 'standard 2D multiplication' do
    # Matrix A: [2, 3]
    let(:a_data) { Numo::SFloat[[1, 2, 3], [4, 5, 6]] }
    # Matrix B: [3, 2]
    let(:b_data) { Numo::SFloat[[7, 8], [9, 10], [11, 12]] }

    let(:a) { Maurograd::Tensor.new(a_data, requires_grad: true) }
    let(:b) { Maurograd::Tensor.new(b_data, requires_grad: true) }
    let(:c) { a.matmul(b) }

    it 'correctly computes the forward pass (C = A . B)' do
      # Expected result: [ [1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12] ]
      expected = Numo::SFloat[[58, 64], [139, 154]]
      expect(c.data).to be == expected
    end

    it 'correctly computes the gradients (backward pass)' do
      # Use an output gradient of all 1s for simplicity.
      grad_output = Numo::SFloat[[1, 1], [1, 1]]
      c.backward(grad_output)

      # grad_a = grad_output . B^T  => [2,2] . [2,3] = [2,3]
      expected_grad_a = grad_output.dot(b_data.transpose)
      expect(a.grad).to be == expected_grad_a

      # grad_b = A^T . grad_output  => [3,2] . [2,2] = [3,2]
      expected_grad_b = a_data.transpose.dot(grad_output)
      expect(b.grad).to be == expected_grad_b
    end
  end

  # # Test 2: Batch MatMul (essential for CNNs).
  # # Numo::NArray handles the dot product across multiple dimensions if the last two axes are compatible.
  # describe 'Batch MatMul' do
  #   it 'correctly handles a batch of matrices' do
  #     # Shape: [Batch=2, M=2, N=2]
  #     a = Maurograd::Tensor.new(Numo::SFloat.new(2, 2, 2).fill(1), requires_grad: true)
  #     b = Maurograd::Tensor.new(Numo::SFloat.new(2, 2, 2).fill(2), requires_grad: true)

  #     c = a.matmul(b)

  #     # Each resulting 2x2 matrix should be [[4, 4], [4, 4]]
  #     expect(c.shape).to be == [2, 2, 2]
  #     expect(c.data[0, true, true]).to be == Numo::SFloat[[4, 4], [4, 4]]
  #   end
  # end
end
