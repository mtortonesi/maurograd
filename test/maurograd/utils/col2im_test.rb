require 'numo/narray'
require_relative '../../../lib/maurograd/utils/utils'

describe Maurograd::Utils do
  describe ".col2im" do
    def ref_im2col(x, fh, fw, stride, padding)
      # Reference, slow but readable: returns [N*OH*OW, C*FH*FW]
      n, c, h, w = x.shape
      oh = (h + 2 * padding - fh) / stride + 1
      ow = (w + 2 * padding - fw) / stride + 1

      # pad manually (same as Utils.pad_nchw)
      if padding > 0
        xp = Numo::SFloat.zeros(n, c, h + 2 * padding, w + 2 * padding)
        xp[true, true, padding...(padding + h), padding...(padding + w)] = x
      else
        xp = x
      end

      rows = Numo::SFloat.zeros(n * oh * ow, c * fh * fw)

      r = 0
      (0...n).each do |bn|
        (0...oh).each do |y|
          (0...ow).each do |xw|
            col = []
            (0...c).each do |cc|
              (0...fh).each do |ky|
                (0...fw).each do |kx|
                  iy = y * stride + ky
                  ix = xw * stride + kx
                  col << xp[bn, cc, iy, ix]
                end
              end
            end
            rows[r, true] = Numo::SFloat.asarray(col)
            r += 1
          end
        end
      end

      rows
    end

    def ref_col2im(col, input_shape, fh, fw, stride, padding)
      n, c, h, w = input_shape
      oh = (h + 2 * padding - fh) / stride + 1
      ow = (w + 2 * padding - fw) / stride + 1

      ph = h + 2 * padding
      pw = w + 2 * padding
      xp = Numo::SFloat.zeros(n, c, ph, pw)

      r = 0
      (0...n).each do |bn|
        (0...oh).each do |y|
          (0...ow).each do |xw|
            idx = 0
            (0...c).each do |cc|
              (0...fh).each do |ky|
                (0...fw).each do |kx|
                  iy = y * stride + ky
                  ix = xw * stride + kx
                  xp[bn, cc, iy, ix] += col[r, idx]
                  idx += 1
                end
              end
            end
            r += 1
          end
        end
      end

      return xp if padding == 0
      xp[true, true, padding...(padding + h), padding...(padding + w)]
    end

    with "no overlap (stride == kernel) and no padding" do
      it "is an exact inverse of im2col (round-trip)" do
        x = Numo::SFloat.new(1, 2, 4, 4).seq.reshape(1, 2, 4, 4) + 1 # deterministic
        fh = 2
        fw = 2
        stride = 2
        padding = 0

        col = Maurograd::Utils.im2col(x, fh, fw, stride, padding)
        x2 = Maurograd::Utils.col2im(col, x.shape, fh, fw, stride, padding)

        expect(x2.shape).to be == x.shape
        expect(x2).to be == x
      end
    end

    with "overlap (stride < kernel) and padding" do
      it "matches a reference implementation (accumulates overlaps correctly)" do
        x = Numo::SFloat[[[
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]
        ]]]

        fh = 2
        fw = 2
        stride = 1
        padding = 1

        # Use the *same* col as the reference (so we're validating col2im logic)
        col = ref_im2col(x, fh, fw, stride, padding)

        got = Maurograd::Utils.col2im(col, x.shape, fh, fw, stride, padding)
        exp = ref_col2im(col, x.shape, fh, fw, stride, padding)

        expect(got.shape).to be == x.shape
        expect(got).to be == exp
      end
    end
  end
end

