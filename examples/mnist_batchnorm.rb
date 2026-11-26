#!/usr/bin/env ruby
# Same LeNet-style CNN as examples/mnist.rb, trained twice - once as-is,
# once with a BatchNorm2D layer after every Conv2D - so you can watch what
# BatchNorm actually changes about training: same architecture, same data,
# same seed, same learning rate, only its presence differs.
#
# What to look for in the output: at the `lr` below, both runs end up in
# roughly the same place by epoch 3 - but the "with BatchNorm2D" run gets
# there much faster. Watch epoch 1's test accuracy specifically: at the time
# this comment was written, "without" landed around 94% after one epoch,
# while "with" was already around 97% - within two points of its own
# epoch-3 result. That's BatchNorm's most reliable practical benefit: not a better
# final answer, but a much shorter path to a good one.
#
# Try varying `lr` yourself. It's tempting to assume BatchNorm always makes
# training *more* forgiving of a high learning rate - that's the textbook
# claim, and it usually holds at the scale a full training run operates at.
# But it's worth checking rather than taking on faith: with plain SGD (no
# momentum) and only a few thousand updates, gamma/beta get pushed by the
# same raw learning rate as everything else, and past a certain point that
# coupling can make the "with BatchNorm2D" run *less* stable than the
# "without" one, not more - the opposite of what the textbook framing
# implies. Push `lr` well past its default and watch for that reversal.

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "maurograd"

Maurograd.seed!(1)

root = "data"
epochs = 3
batch_size = 64
lr = 0.3

Maurograd::Datasets::MNIST.download(root: root)
x_train, y_train = Maurograd::Datasets::MNIST.load(root: root, split: :train, normalize: true)
x_test, y_test = Maurograd::Datasets::MNIST.load(root: root, split: :test, normalize: true)
y_train = Numo::Int32.cast(y_train)

def build_model(batchnorm:)
  layers = [Maurograd::Layers::Conv2D.new(1, 6, 5, padding: 2)]
  layers << Maurograd::Layers::BatchNorm2D.new(6) if batchnorm
  layers += [
    Maurograd::Layers::ReLU.new,
    Maurograd::Layers::MaxPool2D.new(2, stride: 2),
    Maurograd::Layers::Conv2D.new(6, 16, 5)
  ]
  layers << Maurograd::Layers::BatchNorm2D.new(16) if batchnorm
  layers += [
    Maurograd::Layers::ReLU.new,
    Maurograd::Layers::MaxPool2D.new(2, stride: 2),
    Maurograd::Layers::Flatten.new,
    Maurograd::Layers::Linear.new(16 * 5 * 5, 120),
    Maurograd::Layers::ReLU.new,
    Maurograd::Layers::Linear.new(120, 84),
    Maurograd::Layers::ReLU.new,
    Maurograd::Layers::Linear.new(84, 10)
  ]
  Maurograd::Layers::Sequential.new(*layers)
end

# Make sure tensors are contiguous for better performance, as Maurograd
# expects contiguous data for efficient access patterns. If not contiguous,
# we create a copy which will be contiguous.
def to_tensor(data)
  Maurograd::Tensor.new(data.contiguous? ? data : data.copy, requires_grad: false)
end

def train_epoch(model, x, y, optimizer, batch_size:, seed:)
  model.train
  loader = Maurograd::Datasets::Batching.batches(x, y, batch_size: batch_size, shuffle: true, seed: seed)

  total_loss = 0.0
  batches = 0

  loader.each do |xb, yb|
    optimizer.zero_grad

    logits = model.forward(to_tensor(xb))
    loss = Maurograd::Losses::CrossEntropyWithLogits.apply(logits, to_tensor(yb))
    loss.backward

    optimizer.step

    total_loss += loss.item
    batches += 1
  end

  total_loss / batches
end

def evaluate(model, x, y, batch_size:)
  model.eval
  loader = Maurograd::Datasets::Batching.batches(x, y, batch_size: batch_size, shuffle: false, seed: 0)

  correct = 0
  total = 0

  loader.each do |xb, yb|
    xb_t = to_tensor(xb)
    yb_t = to_tensor(yb)

    logits = model.forward(xb_t)
    pred = logits.to_numo.max_index(axis: 1) % 10

    correct += pred.eq(yb_t.to_numo).count_true
    total += yb_t.shape[0]
  end

  100.0 * correct / total
end

def run(name, batchnorm:, x_train:, y_train:, x_test:, y_test:, epochs:, batch_size:, lr:)
  puts "== #{name} =="

  model = build_model(batchnorm: batchnorm)
  optimizer = Maurograd::Optim::SGD.new(model.parameters, lr: lr)

  1.upto(epochs) do |e|
    avg_loss = train_epoch(model, x_train, y_train, optimizer, batch_size: batch_size, seed: e)
    accuracy = evaluate(model, x_test, y_test, batch_size: 256)
    puts "  epoch #{e} | avg train loss #{avg_loss.round(4)} | test accuracy #{accuracy.round(2)}%"
  end

  puts
end

run(
  "without BatchNorm2D", batchnorm: false,
  x_train: x_train, y_train: y_train, x_test: x_test, y_test: y_test,
  epochs: epochs, batch_size: batch_size, lr: lr
)

run(
  "with BatchNorm2D", batchnorm: true,
  x_train: x_train, y_train: y_train, x_test: x_test, y_test: y_test,
  epochs: epochs, batch_size: batch_size, lr: lr
)
