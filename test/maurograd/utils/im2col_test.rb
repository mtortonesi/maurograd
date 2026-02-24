# test/maurograd/utils/im2col_test.rb

require 'numo/narray'
require_relative '../../../lib/maurograd/utils/utils'

describe Maurograd::Utils do
  describe ".im2col" do
    with "stride > 1 and no padding (no overlap)" do
      it "extracts receptive fields in N,OH,OW row order" do
        # Input: N=1, C=1, H=4, W=4
        # Values:
        #  1  2  3  4
        #  5  6  7  8
        #  9 10 11 12
        # 13 14 15 16
        x = Numo::SFloat[[[
          [ 1,  2,  3,  4],
          [ 5,  6,  7,  8],
          [ 9, 10, 11, 12],
          [13, 14, 15, 16]
        ]]]

        fh = 2
        fw = 2
        stride = 2
        padding = 0

        col = Maurograd::Utils.im2col(x, fh, fw, stride, padding)

        # out_h = (4 - 2) / 2 + 1 = 2
        # out_w = (4 - 2) / 2 + 1 = 2
        # Rows = N * OH * OW = 4
        # Cols = C * FH * FW = 4
        expect(col.shape).to be == [4, 4]

        # Expected patches (row-major within each patch):
        # (oh=0, ow=0) -> [[1,2],[5,6]] => [1,2,5,6]
        # (oh=0, ow=1) -> [[3,4],[7,8]] => [3,4,7,8]
        # (oh=1, ow=0) -> [[9,10],[13,14]] => [9,10,13,14]
        # (oh=1, ow=1) -> [[11,12],[15,16]] => [11,12,15,16]
        expected = Numo::SFloat[
          [ 1,  2,  5,  6],
          [ 3,  4,  7,  8],
          [ 9, 10, 13, 14],
          [11, 12, 15, 16]
        ]

        expect(col).to be == expected
      end
    end

    with "padding = 1 and stride = 1" do
      it "includes zero padding correctly and produces expected shape" do
        # Input: N=1, C=1, H=2, W=2
        # 1 2
        # 3 4
        x = Numo::SFloat[[[
          [1, 2],
          [3, 4]
        ]]]

        fh = 2
        fw = 2
        stride = 1
        padding = 1

        col = Maurograd::Utils.im2col(x, fh, fw, stride, padding)

        # Padded H,W become 4x4
        # out_h = (2 + 2*1 - 2)/1 + 1 = 3
        # out_w = (2 + 2*1 - 2)/1 + 1 = 3
        expect(col.shape).to be == [9, 4]

        # We check a few representative rows.
        # Row order is (oh, ow):
        #
        # (0,0): patch over padded top-left corner:
        # [0 0
        #  0 1] -> [0,0,0,1]
        expect(col[0, true]).to be == Numo::SFloat[0, 0, 0, 1]

        # (0,1): patch:
        # [0 0
        #  1 2] -> [0,0,1,2]
        expect(col[1, true]).to be == Numo::SFloat[0, 0, 1, 2]

        # (1,0): patch:
        # [0 1
        #  0 3] -> [0,1,0,3]
        expect(col[3, true]).to be == Numo::SFloat[0, 1, 0, 3]

        # (1,1): patch fully inside original data:
        # [1 2
        #  3 4] -> [1,2,3,4]
        expect(col[4, true]).to be == Numo::SFloat[1, 2, 3, 4]

        # (2,2): bottom-right corner:
        # [4 0
        #  0 0] -> [4,0,0,0]
        expect(col[8, true]).to be == Numo::SFloat[4, 0, 0, 0]
      end
    end

    with "1x1 kernel (identity-like extraction)" do
      it "extracts each pixel into its own row when stride=1 and padding=0" do
        # Input: N=1, C=1, H=2, W=3
        x = Numo::SFloat[[[
          [1, 2, 3],
          [4, 5, 6]
        ]]]

        fh = 1
        fw = 1
        stride = 1
        padding = 0

        col = Maurograd::Utils.im2col(x, fh, fw, stride, padding)

        # out_h = 2, out_w = 3 => rows=6, cols=1
        expect(col.shape).to be == [6, 1]

        expected = Numo::SFloat[
          [1],
          [2],
          [3],
          [4],
          [5],
          [6]
        ]
        expect(col).to be == expected
      end
    end
  end
end

