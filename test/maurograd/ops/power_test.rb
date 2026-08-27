describe Maurograd::Ops::Power do
  it 'should correctly compute a simple loss function' do
    # Assuming the necessary files are loaded
    x = Maurograd::Tensor.new(Numo::SFloat[2, 3], requires_grad: true)

    # Forward pass: L = sum(x^2)
    x_squared = Maurograd::Ops::Power.apply(x, 2)
    loss = Maurograd::Ops::Sum.apply(x_squared)

    expect(loss.data).to be == 13.0

    # Backward pass
    loss.backward

    expect(x.grad).to be == Numo::SFloat[4.0, 6.0]
  end
end
