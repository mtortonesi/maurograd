describe Maurograd::Utils do
  describe '.im2col' do
    let(:input) do
      # Creiamo un'immagine 1x3x3: [N=1, C=1, H=3, W=3]
      Numo::SFloat[[[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
      ]]]
    end


    # Data l'immagine input e un filtro 2×2, le finestre estratte dovrebbero essere:
    # - In alto a sinistra: [[1, 2], [4, 5]]
    # - In alto a destra: [[2, 3], [5, 6]]
    # - In basso a sinistra: [[4, 5], [7, 8]]
    # - In basso a destra: [[5, 6], [8, 9]]
    # La funzione im2col dovrebbe appiattire queste finestre in righe di una matrice.
    it 'trasforma correttamente una piccola immagine in colonne' do
      # Parametri: filter_h=2, filter_w=2, stride=1, padding=0
      res = Maurograd::Utils.im2col(input, 2, 2, 1, 0)

      # puts "res.shape = #{res.shape}"
      # puts "res[0, true] = #{res[0, true].inspect}"

      # Con un'immagine 3x3 e filtro 2x2, l'output spaziale è 2x2.
      # La matrice finale deve avere (N * out_h * out_w) righe = 4
      # e (C * fh * fw) colonne = 4
      expect(res.shape).to be == [4, 4]

      # Verifichiamo la prima riga (prima finestra: [[1, 2], [4, 5]])
      expect(res[0, true]).to be == Numo::SFloat[1, 2, 4, 5]

      # Verifichiamo l'ultima riga (ultima finestra: [[5, 6], [8, 9]])
      expect(res[3, true]).to be == Numo::SFloat[5, 6, 8, 9]
    end

    it 'gestisce correttamente lo stride' do
      # Con stride 2, su una 3x3 con filtro 2x2, rimane solo la prima finestra
      res = Maurograd::Utils.im2col(input, 2, 2, 2, 0)

      expect(res.shape).to be == [1, 4]
      expect(res[0, true]).to be == Numo::SFloat[1, 2, 4, 5]
    end
  end

  describe '.col2im' do
    it "ricostruisce correttamente le zone di sovrapposizione" do
      shape = [1, 1, 3, 3]
      input = Numo::SFloat.ones(*shape)

      # im2col
      col = Maurograd::Utils.im2col(input, 2, 2, 1, 0)

      # col2im
      res = Maurograd::Utils.col2im(col, shape, 2, 2, 1, 0)

      # Il pixel centrale [1,1] è coperto da 4 finestre 2x2
      # Gli angoli sono coperti da 1 sola finestra
      expected = Numo::SFloat[[[[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]]]]

      expect(res).to be == expected
    end
  end
end
