describe Maurograd::Utils do
  describe '.im2col' do
    let(:input) do
      # Create a 1x1x3x3 image: [N=1, C=1, H=3, W=3]
      Numo::SFloat[[[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
      ]]]
    end


    # Given the input image and a 2x2 filter, the extracted windows should be:
    # - top-left: [[1, 2], [4, 5]]
    # - top-right: [[2, 3], [5, 6]]
    # - bottom-left: [[4, 5], [7, 8]]
    # - bottom-right: [[5, 6], [8, 9]]
    # im2col should flatten these windows into the rows of a matrix.
    it 'correctly turns a small image into columns' do
      # Parameters: filter_h=2, filter_w=2, stride=1, padding=0
      res = Maurograd::Utils.im2col(input, 2, 2, 1, 0)

      # With a 3x3 image and a 2x2 filter, the spatial output is 2x2.
      # The final matrix must have (N * out_h * out_w) rows = 4
      # and (C * fh * fw) columns = 4.
      expect(res.shape).to be == [4, 4]

      # Check the first row (first window: [[1, 2], [4, 5]]).
      expect(res[0, true]).to be == Numo::SFloat[1, 2, 4, 5]

      # Check the last row (last window: [[5, 6], [8, 9]]).
      expect(res[3, true]).to be == Numo::SFloat[5, 6, 8, 9]
    end

    it 'correctly handles the stride' do
      # With stride 2, on a 3x3 image with a 2x2 filter, only the first window remains.
      res = Maurograd::Utils.im2col(input, 2, 2, 2, 0)

      expect(res.shape).to be == [1, 4]
      expect(res[0, true]).to be == Numo::SFloat[1, 2, 4, 5]
    end
  end

  describe '.col2im' do
    it "correctly reconstructs overlapping regions" do
      shape = [1, 1, 3, 3]
      input = Numo::SFloat.ones(*shape)

      # im2col
      col = Maurograd::Utils.im2col(input, 2, 2, 1, 0)

      # col2im
      res = Maurograd::Utils.col2im(col, shape, 2, 2, 1, 0)

      # The center pixel [1,1] is covered by 4 overlapping 2x2 windows.
      # The corners are covered by a single window.
      expected = Numo::SFloat[[[[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]]]]

      expect(res).to be == expected
    end
  end
end
