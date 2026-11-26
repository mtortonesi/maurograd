# Smoke test for the *installed* gems, run by CI after building and
# installing them (and runnable by hand): loads Maurograd outside the repo,
# checks it really came from an installed gem, and trains one step.
#
#   ruby script/smoke_installed_gem.rb        # maurograd, CPU backend
#   ruby script/smoke_installed_gem.rb mlx    # maurograd-mlx, MLX backend
#
# Kept out of test/: sus loads every .rb file there.
backend = (ARGV[0] || "cpu").to_sym

require "maurograd"

loaded_from = $LOADED_FEATURES.find { |f| f.end_with?("/maurograd.rb") }
unless loaded_from&.include?("/gems/maurograd-")
  abort "maurograd was loaded from #{loaded_from.inspect}, not from an installed gem"
end

Maurograd.backend = backend
Maurograd.seed!(1)

model = Maurograd::Layers::Sequential.new(
  Maurograd::Layers::Conv2D.new(1, 2, 3), Maurograd::Layers::BatchNorm2D.new(2),
  Maurograd::Layers::ReLU.new, Maurograd::Layers::MaxPool2D.new(2),
  Maurograd::Layers::Flatten.new, Maurograd::Layers::Linear.new(2 * 3 * 3, 3)
)
optimizer = Maurograd::Optim::Adam.new(model.parameters, lr: 1e-2)
x = Maurograd::Tensor.new(Numo::SFloat.new(4, 1, 8, 8).rand)
y = Maurograd::Tensor.new(Numo::Int32[0, 1, 2, 0])

losses = Array.new(5) do
  optimizer.zero_grad
  loss = Maurograd::Losses::CrossEntropyWithLogits.apply(model.forward(x), y)
  loss.backward
  optimizer.step
  loss.item
end

abort "loss did not decrease: #{losses.inspect}" unless losses.last < losses.first

puts "maurograd #{Maurograd::VERSION} (#{loaded_from}) on #{backend}: " \
  "loss #{losses.first.round(4)} -> #{losses.last.round(4)}"
