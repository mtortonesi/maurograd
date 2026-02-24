describe Maurograd::Layers::Conv2D do
  with "a single channel input and output" do
    let(:in_channels) { 1 }
    let(:out_channels) { 1 }
    let(:kernel_size) { 2 }
    let(:stride) { 1 }
    let(:padding) { 0 }

    it "correcty applies forward pass with unit weights" do
      conv = Maurograd::Layers::Conv2D.new(in_channels, out_channels, kernel_size, stride: stride, padding: padding)

      # Sovrascriviamo i pesi casuali con pesi tutti a 1.0 e bias a 0.5
      # Forma pesi: [out_f, in_c, kh, kw] -> [1, 1, 2, 2]
      # conv.weights.data = Numo::SFloat.ones(1, 1, 2, 2)
      conv.weights.data = Numo::SFloat.ones(out_channels, in_channels, 2, 2)
      conv.bias.data = Numo::SFloat[0.5]

      # Forma input: [batch_size, channels, height, width]
      # conv.weights.data = Numo::SFloat.ones(1, 1, 2, 2)
      # Input 1x1x3x3
      # [[[1, 2, 3],
      #   [4, 5, 6],
      #   [7, 8, 9]]]
      input_data = Numo::SFloat[[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
      input = Maurograd::Tensor.new(input_data)

      output = conv.forward(input)

      # Calcolo manuale (somma della finestra 2x2 + bias 0.5):
      # Finestra 1: (1+2+4+5) + 0.5 = 12.5
      # Finestra 2: (2+3+5+6) + 0.5 = 16.5
      # Finestra 3: (4+5+7+8) + 0.5 = 24.5
      # Finestra 4: (5+6+8+9) + 0.5 = 28.5
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

      # Impostiamo pesi tutti a 1.0 per ogni filtro e ogni canale di ingresso
      # Forma: [8, 3, 2, 2]
      conv.weights.data = Numo::SFloat.ones(out_channels, in_channels, 2, 2)
      # Impostiamo il bias a 0.5 per tutti gli 8 filtri in uscita
      conv.bias.data = Numo::SFloat.new(out_channels).fill(0.5)

      # Input 1x3x3x3 con valori identici su ogni canale per facilitare i conti
      single_channel = Numo::SFloat[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
      input_data = Numo::SFloat.zeros(1, 3, 3, 3)
      (0...3).each { |c| input_data[0, c, true, true] = single_channel }

      input = Maurograd::Tensor.new(input_data)
      output = conv.forward(input)

      # Verifica Shape: [Batch, Out_Channels, Out_H, Out_W]
      expect(output.shape).to be == [1, 8, 2, 2]

      # Verifica Valori: 
      # Ogni finestra di ogni canale vale 12, 16, 24, 28.
      # Essendoci 3 canali sommati: (Valore * 3) + 0.5
      # 12*3 + 0.5 = 36.5
      # 16*3 + 0.5 = 48.5
      # 24*3 + 0.5 = 72.5
      # 28*3 + 0.5 = 84.5

      expected_one_filter = Numo::SFloat[[36.5, 48.5], [72.5, 84.5]]

      # Controlliamo che il primo degli 8 filtri sia corretto
      expect(output.data[0, 0, true, true]).to be == expected_one_filter

      # Controlliamo l'ultimo degli 8 filtri (dovrebbe essere uguale nel nostro test)
      expect(output.data[0, 7, true, true]).to be == expected_one_filter
    end
  end

  # with "multiple input and output channels" do
  #   let(:in_channels) { 3 }
  #   let(:out_channels) { 8 }
  #   let(:kernel_size) { 2 }
  #   let(:stride) { 1 }
  #   let(:padding) { 0 }

    # it "correcty applies forward pass with unit weights" do
    #   conv = Maurograd::Layers::Conv2D.new(in_channels, out_channels, kernel_size)

    #   # In una CNN, i pesi non sono solo una matrice, ma un tensore 4D con forma:
    #   # [F,C,FH,FW], dove:
    #   #
    #   # - F: Numero di filtri in uscita (out_channels).
    #   # - C: Numero di canali in ingresso (in_channels).
    #   # - FH,FW: Altezza e larghezza del filtro (es. 2x2).

    #   # Sovrascriviamo i pesi casuali con pesi tutti a 1.0 e bias a 0.5
    #   # Forma pesi: [out_f, in_c, kh, kw] -> [1, 1, 2, 2]
    #   conv.weights.data = Numo::SFloat[[[[1, 1],
    #                                      [1, 1]],
    #                                     [[1, 1],
    #                                      [1, 1]],
    #                                     [[1, 1],
    #                                      [1, 1]],
    #                                     [[1, 1],
    #                                      [1, 1]],
    #                                     [[1, 1],
    #                                      [1, 1]],
    #                                     [[1, 1],
    #                                      [1, 1]],
    #                                     [[1, 1],
    #                                      [1, 1]],
    #                                     [[1, 1],
    #                                      [1, 1]]],
    #                                    [[[2, 2],
    #                                      [2, 2]],
    #                                     [[2, 2],
    #                                      [2, 2]],
    #                                     [[2, 2],
    #                                      [2, 2]],
    #                                     [[2, 2],
    #                                      [2, 2]],
    #                                     [[2, 2],
    #                                      [2, 2]],
    #                                     [[2, 2],
    #                                      [2, 2]],
    #                                     [[2, 2],
    #                                      [2, 2]],
    #                                     [[2, 2],
    #                                      [2, 2]]],
    #                                    [[[3, 3],
    #                                      [3, 3]],
    #                                     [[3, 3],
    #                                      [3, 3]],
    #                                     [[3, 3],
    #                                      [3, 3]],
    #                                     [[3, 3],
    #                                      [3, 3]],
    #                                     [[3, 3],
    #                                      [3, 3]],
    #                                     [[3, 3],
    #                                      [3, 3]],
    #                                     [[3, 3],
    #                                      [3, 3]],
    #                                     [[3, 3],
    #                                      [3, 3]]],
    #                                    [[[4, 4],
    #                                      [4, 4]],
    #                                     [[4, 4],
    #                                      [4, 4]],
    #                                     [[4, 4],
    #                                      [4, 4]],
    #                                     [[4, 4],
    #                                      [4, 4]],
    #                                     [[4, 4],
    #                                      [4, 4]],
    #                                     [[4, 4],
    #                                      [4, 4]],
    #                                     [[4, 4],
    #                                      [4, 4]],
    #                                     [[4, 4],
    #                                      [4, 4]]],
    #                                    [[[5, 5],
    #                                      [5, 5]],
    #                                     [[5, 5],
    #                                      [5, 5]],
    #                                     [[5, 5],
    #                                      [5, 5]],
    #                                     [[5, 5],
    #                                      [5, 5]],
    #                                     [[5, 5],
    #                                      [5, 5]],
    #                                     [[5, 5],
    #                                      [5, 5]],
    #                                     [[5, 5],
    #                                      [5, 5]],
    #                                     [[5, 5],
    #                                      [5, 5]]],
    #                                    [[[6, 6],
    #                                      [6, 6]],
    #                                     [[6, 6],
    #                                      [6, 6]],
    #                                     [[6, 6],
    #                                      [6, 6]],
    #                                     [[6, 6],
    #                                      [6, 6]],
    #                                     [[6, 6],
    #                                      [6, 6]],
    #                                     [[6, 6],
    #                                      [6, 6]],
    #                                     [[6, 6],
    #                                      [6, 6]],
    #                                     [[6, 6],
    #                                      [6, 6]]],
    #                                    [[[7, 7],
    #                                      [7, 7]],
    #                                     [[7, 7],
    #                                      [7, 7]],
    #                                     [[7, 7],
    #                                      [7, 7]],
    #                                     [[7, 7],
    #                                      [7, 7]],
    #                                     [[7, 7],
    #                                      [7, 7]],
    #                                     [[7, 7],
    #                                      [7, 7]],
    #                                     [[7, 7],
    #                                      [7, 7]],
    #                                     [[7, 7],
    #                                      [7, 7]]],
    #                                    [[[8, 8],
    #                                      [8, 8]],
    #                                     [[8, 8],
    #                                      [8, 8]],
    #                                     [[8, 8],
    #                                      [8, 8]],
    #                                     [[8, 8],
    #                                      [8, 8]],
    #                                     [[8, 8],
    #                                      [8, 8]],
    #                                     [[8, 8],
    #                                      [8, 8]],
    #                                     [[8, 8],
    #                                      [8, 8]],
    #                                     [[8, 8],
    #                                      [8, 8]]]]
    #   conv.bias.data = Numo::SFloat[0.5]

    #   # Forma input: [batch_size, channels, height, width]
    #   # Input 1x3x3x3
    #   input_data = Numo::SFloat[[[[1, 2, 3],
    #                               [4, 5, 6],
    #                               [7, 8, 9]],
    #                              [[1, 2, 3],
    #                               [4, 5, 6],
    #                               [7, 8, 9]],
    #                              [[1, 2, 3],
    #                               [4, 5, 6],
    #                               [7, 8, 9]]]]
    #   input = Maurograd::Tensor.new(input_data)
    #   # puts "input shape: #{input.shape.inspect}"

    #   output = conv.forward(input)

    #   # Calcolo manuale (somma della finestra 2x2 + bias 0.5):
    #   # Finestra 1: (1+2+4+5) + 0.5 = 12.5
    #   # Finestra 2: (2+3+5+6) + 0.5 = 16.5
    #   # Finestra 3: (4+5+7+8) + 0.5 = 24.5
    #   # Finestra 4: (5+6+8+9) + 0.5 = 28.5
    #   expected = Numo::SFloat[[[[12.5, 16.5], [24.5, 28.5]]]]

    #   # Nelle CNN, la convoluzione "consuma" la dimensione dei canali di input. Se hai 3 canali di input e applichi dei filtri, ogni filtro guarda tutti e 3 i canali contemporaneamente e somma i risultati in un unico valore scalare per ogni posizione spaziale.

    # # Input: [N=1,C=3,H=3,W=3]

    # # Filtri: [F=8,C=3,FH=2,FW=2]

    # # Output atteso: [N=1,F=8,OH=2,OW=2]
    #   expect(output.shape).to be == [1, out_channels, 2, 2]
    #   # expect(output.data).to be == expected
    # end

  # end
end
