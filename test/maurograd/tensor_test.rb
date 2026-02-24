describe Maurograd::Tensor do
  let(:data) { Numo::SFloat.new(2, 2).seq }
  let(:tensor) { Maurograd::Tensor.new(data) }

  it 'ha la forma corretta' do
    expect(tensor.shape).to be == [2, 2]
  end

  # it 'supporta requires_grad' do
  #   t = Maurograd::Tensor.new(data, requires_grad: true)
  #   expect(t.grad).not.to be_nil
  # end

  # describe "Trasposizione Tensori Multidimensionali" do
  #   it "verifica il comportamento di transpose su tensori 3D" do
  #     # Creiamo un tensore [2, 3, 4] -> Batch=2, Righe=3, Colonne=4
  #     data = Numo::SFloat.new(2, 3, 4).seq

  #     # Transpose standard di Numo inverte tutto -> [4, 3, 2]
  #     expect(data.transpose.shape).to be == [4, 3, 2]

  #     # Per la MatMul noi vogliamo invertire solo gli ultimi due -> [2, 4, 3]
  #     # In Numo si fa passando gli indici degli assi: [0, 2, 1]
  #     batch_transpose = data.transpose(0, 2, 1)
  #     expect(batch_transpose.shape).to be == [2, 4, 3]
  #   end
  # end

  # Vogliamo verificare che la trasposizione dei tensori multidimensionali
  # inverta solo le ultime due dimensioni (H e W) mantenendo intatte le altre (N e C)
  describe "Manipolazione Tensori 4D" do
    it "inverte correttamente solo le dimensioni spaziali H e W" do
      # Forma: [N=2, C=3, H=4, W=5]
      data = Numo::SFloat.new(2, 3, 4, 5).seq
      
      # Vogliamo passare da [0, 1, 2, 3] a [0, 1, 3, 2]
      # Questo scambia H e W mantenendo intatti N e C
      res = data.transpose(0, 1, 3, 2)
      
      expect(res.shape).to be == [2, 3, 5, 4]
    end
  end

  # Test della Somma Semplice (senza broadcasting)
  describe 'operazione di somma semplice' do
    let(:a) { Maurograd::Tensor.new(Numo::SFloat[1, 2, 3], requires_grad: true) }
    let(:b) { Maurograd::Tensor.new(Numo::SFloat[4, 5, 6], requires_grad: true) }
    let(:c) { a + b }

    it 'calcola correttamente il forward pass' do
      expect(c.data).to be == Numo::SFloat[5, 7, 9]
    end

    it 'accumula correttamente i gradienti quando un tensore è usato più volte' do
      a = Maurograd::Tensor.new(Numo::SFloat[2, 3], requires_grad: true)
      # Operazione: b = a + a
      b = a + a

      b.backward(Numo::SFloat[1, 1])

      # La derivata di x + x è 2. Quindi il gradiente deve essere [2, 2]
      expect(a.grad).to be == Numo::SFloat[2, 2]
    end

    it 'calcola correttamente i gradienti nel backward pass' do
      c.backward(Numo::SFloat[1, 1, 1])

      expect(a.grad).to be == Numo::SFloat[1, 1, 1]
      expect(b.grad).to be == Numo::SFloat[1, 1, 1]
    end
  end

  # Test del Broadcasting (fondamentale per i Bias delle CNN)
  describe 'broadcasting del gradiente' do
    # Immaginiamo un mini-batch di 2 campioni, con 3 canali (es. 2x3)
    let(:data) { Numo::SFloat[[1, 2, 3], [4, 5, 6]] }
    let(:bias_data) { Numo::SFloat[[10, 20, 30]] } # Forma [1, 3]

    let(:a) { Maurograd::Tensor.new(data, requires_grad: true) }
    let(:b) { Maurograd::Tensor.new(bias_data, requires_grad: true) }
    let(:c) { a + b }

    it 'esegue correttamente il broadcasting nel forward' do
      expected = Numo::SFloat[[11, 22, 33], [14, 25, 36]]
      expect(c.data).to be == expected
    end

    it 'riduce correttamente il gradiente per il tensore broadcasted' do
      # Gradiente in uscita (stessa forma di c: 2x3)
      grad_output = Numo::SFloat[[1, 1, 1], [1, 1, 1]]
      c.backward(grad_output)

      # Per 'a', il gradiente deve essere identico a grad_output (2x3)
      expect(a.grad).to be == grad_output

      # Per 'b' (il bias), il gradiente deve essere la somma lungo l'asse del batch
      # Risultato atteso: [1+1, 1+1, 1+1] -> [2, 2, 2] con forma [1, 3]
      expect(b.grad.shape).to be == [1, 3]
      expect(b.grad).to be == Numo::SFloat[[2, 2, 2]]
    end
  end
end
