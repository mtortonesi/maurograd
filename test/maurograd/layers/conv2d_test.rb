describe Maurograd::Layers::Conv2D do
  with "a single channel input and output" do
    let(:in_channels) { 1 }
    let(:out_channels) { 1 }
    let(:kernel_size) { 2 }
    let(:stride) { 1 }
    let(:padding) { 0 }

    it "correcty applies forward pass with unit weights" do
      conv = Maurograd::Layers::Conv2D.new(in_channels, out_channels, kernel_size, stride: stride, padding: padding)

      # Overwrite the random weights with weights all equal to 1.0 and bias 0.5.
      # Weight shape: [out_f, in_c, kh, kw] -> [1, 1, 2, 2]
      conv.weights.data = Numo::SFloat.ones(out_channels, in_channels, 2, 2)
      conv.bias.data = Numo::SFloat[0.5]

      # Input shape: [batch_size, channels, height, width]
      # Input 1x1x3x3
      # [[[1, 2, 3],
      #   [4, 5, 6],
      #   [7, 8, 9]]]
      input_data = Numo::SFloat[[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
      input = Maurograd::Tensor.new(input_data)

      output = conv.forward(input)

      # Manual computation (sum of the 2x2 window + bias 0.5):
      # Window 1: (1+2+4+5) + 0.5 = 12.5
      # Window 2: (2+3+5+6) + 0.5 = 16.5
      # Window 3: (4+5+7+8) + 0.5 = 24.5
      # Window 4: (5+6+8+9) + 0.5 = 28.5
      expected = Numo::SFloat[[[[12.5, 16.5], [24.5, 28.5]]]]

      expect(output.shape).to be == [1, 1, 2, 2]
      expect(output.data).to be == expected
    end
  end

  with "multiple input and output channels" do
    let(:in_channels) { 3 }
    let(:out_channels) { 8 }
    let(:kernel_size) { 2 }

    it "correctly applies forward pass and collapses input channels" do
      conv = Maurograd::Layers::Conv2D.new(in_channels, out_channels, kernel_size)

      # Set every filter's weights to 1.0 for every input channel.
      # Shape: [8, 3, 2, 2]
      conv.weights.data = Numo::SFloat.ones(out_channels, in_channels, 2, 2)
      # Set the bias to 0.5 for all 8 output filters.
      conv.bias.data = Numo::SFloat.new(out_channels).fill(0.5)

      # Input 1x3x3x3 with identical values on every channel, to keep the math simple.
      single_channel = Numo::SFloat[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
      input_data = Numo::SFloat.zeros(1, 3, 3, 3)
      (0...3).each { |c| input_data[0, c, true, true] = single_channel }

      input = Maurograd::Tensor.new(input_data)
      output = conv.forward(input)

      # Check shape: [Batch, Out_Channels, Out_H, Out_W]
      expect(output.shape).to be == [1, 8, 2, 2]

      # Check values:
      # Each channel's window is worth 12, 16, 24, 28.
      # With 3 channels summed: (value * 3) + 0.5
      # 12*3 + 0.5 = 36.5
      # 16*3 + 0.5 = 48.5
      # 24*3 + 0.5 = 72.5
      # 28*3 + 0.5 = 84.5

      expected_one_filter = Numo::SFloat[[36.5, 48.5], [72.5, 84.5]]

      # Check that the first of the 8 filters is correct.
      expect(output.data[0, 0, true, true]).to be == expected_one_filter

      # Check that the last of the 8 filters is correct too (should match in this test).
      expect(output.data[0, 7, true, true]).to be == expected_one_filter
    end
  end
end
