require 'maurograd/tensor'
require 'maurograd/layers/conv2d'

describe "Conv2D Gradient Check" do
  it "ha un gradiente analitico che corrisponde a quello numerico" do
    epsilon = 1e-4
    tolerance = 1e-5

    # 1. Setup: Piccola convoluzione per velocità
    conv = Maurograd::Layers::Conv2D.new(1, 1, 2, stride: 1, padding: 0)
    input = Maurograd::Tensor.new(Numo::SFloat.new(1, 1, 3, 3).rand, requires_grad: true)

    # 2. Forward e Backward analitico
    output = conv.forward(input)

    # Usiamo la somma dei quadrati come funzione di Loss scalare
    loss = (output**2).sum
    loss.backward

    # Salviamo il gradiente analitico calcolato dal tuo framework
    # Ne controlliamo solo uno per brevità (es. il peso [0,0,0,0])
    grad_analitico = conv.weights.grad[0, 0, 0, 0]

    # 3. Calcolo Gradiente Numerico
    original_weight = conv.weights.data[0, 0, 0, 0]

    # Loss(W + epsilon)
    conv.weights.data[0, 0, 0, 0] = original_weight + epsilon
    out_plus = conv.forward(input)
    loss_plus = (out_plus**2).sum.data

    # Loss(W - epsilon)
    conv.weights.data[0, 0, 0, 0] = original_weight - epsilon
    out_minus = conv.forward(input)
    loss_minus = (out_minus**2).sum.data

    grad_numerico = (loss_plus - loss_minus) / (2 * epsilon)

    # 4. Confronto
    diff = (grad_analitico - grad_numerico).abs
    # puts "Analitico: #{grad_analitico}, Numerico: #{grad_numerico.to_f}, Diff: #{diff.to_f}"

    expect(diff).to be < tolerance
  end
end
