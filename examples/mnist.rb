#!/usr/bin/env ruby
$LOAD_PATH.unshift File.expand_path('../lib', __dir__)
require 'maurograd'
require 'stackprof'

root = 'data'
epochs = 5
batch_size = 64
seed = 1

# Download (idempotent) and load.
Maurograd::Datasets::MNIST.download(root: root)
x_train, y_train = Maurograd::Datasets::MNIST.load(root: root, split: :train, normalize: true)
x_test,  y_test  = Maurograd::Datasets::MNIST.load(root: root, split: :test,  normalize: true)
y_train = Numo::Int32.cast(y_train)

model = Maurograd::Layers::Sequential.new(
  Maurograd::Layers::Conv2D.new(1, 6, 5, padding: 2),
  Maurograd::Layers::ReLU.new,
  Maurograd::Layers::MaxPool2D.new(2, stride: 2),

  Maurograd::Layers::Conv2D.new(6, 16, 5),
  Maurograd::Layers::ReLU.new,
  Maurograd::Layers::MaxPool2D.new(2, stride: 2),

  Maurograd::Layers::Flatten.new,
  Maurograd::Layers::Linear.new(16 * 5 * 5, 120),
  Maurograd::Layers::ReLU.new,
  Maurograd::Layers::Linear.new(120, 84),
  Maurograd::Layers::ReLU.new,
  Maurograd::Layers::Linear.new(84, 10)
)

optimizer = Maurograd::Optim::SGD.new(model.parameters, lr: 0.01)

def train_epoch(model, x, y, optimizer, batch_size:, seed:, epoch:)
  model.train
  loader = Maurograd::Datasets::Batching.batches(
    x, y,
    batch_size: batch_size,
    shuffle: true,
    seed: seed # seed + epoch  # to vary the seed on every epoch
  )

  i = 0
  tb0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  loss = nil

  if epoch == 1
    puts 'Running profiler on first epoch...'
    StackProf.run(mode: :wall, out: 'stackprof.dump') do
      loader.each do |xb, yb|
        optimizer.zero_grad

        # Make sure tensors are contiguous for better performance, as Maurograd
        # expects contiguous data for efficient access patterns. If not contiguous,
        # we create a copy which will be contiguous.
        xb_t = Maurograd::Tensor.new(
          xb.contiguous? ? xb : xb.copy,
          requires_grad: false
        )
        yb_t = Maurograd::Tensor.new(
          yb.contiguous? ? yb : yb.copy,
          requires_grad: false
        )

        logits = model.forward(xb_t)
        loss = Maurograd::Losses::CrossEntropyWithLogits.apply(logits, yb_t)
        loss.backward

        optimizer.step

        if (i % 10).zero?
          tb1 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
          puts "Epoch #{epoch} | batch #{i} | dt=#{(tb1 - tb0).round(2)}s | loss #{loss.data.to_f}"
          tb0 = tb1
        end
        i += 1
      end
    end

    puts 'Profiler run complete. Output written to stackprof.dump'
  else
    loader.each do |xb, yb|
      optimizer.zero_grad

      # Make sure tensors are contiguous for better performance, as Maurograd
      # expects contiguous data for efficient access patterns. If not contiguous,
      # we create a copy which will be contiguous.
      xb_t = Maurograd::Tensor.new(
        xb.contiguous? ? xb : xb.copy,
        requires_grad: false
      )
      yb_t = Maurograd::Tensor.new(
        yb.contiguous? ? yb : yb.copy,
        requires_grad: false
      )

      logits = model.forward(xb_t)
      loss = Maurograd::Losses::CrossEntropyWithLogits.apply(logits, yb_t)
      loss.backward

      optimizer.step

      if (i % 10).zero?
        tb1 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        puts "Epoch #{epoch} | batch #{i} | dt=#{(tb1 - tb0).round(2)}s | loss #{loss.data.to_f}"
        tb0 = tb1
      end
      i += 1
    end
  end
end


def evaluate(model, x, y, batch_size:)
  model.eval
  loader = Maurograd::Datasets::Batching.batches(x, y, batch_size: batch_size, shuffle: false, seed: 0)

  correct = 0
  total = 0

  loader.each do |xb, yb|
    # Make sure tensors are contiguous for better performance, as Maurograd
    # expects contiguous data for efficient access patterns. If not contiguous,
    # we create a copy which will be contiguous.
    xb_e = Maurograd::Tensor.new(
      xb.contiguous? ? xb : xb.copy,
      requires_grad: false
    )
    yb_e = Maurograd::Tensor.new(
      yb.contiguous? ? yb : yb.copy,
      requires_grad: false
    )
    logits = model.forward(xb_e)
    pred = logits.data.max_index(axis: 1) % 10

    correct += (pred.eq(yb_e.data)).count_true
    total += yb_e.shape[0]
  end

  puts "Accuracy: #{(100.0 * correct / total).round(2)}%"
end

1.upto(epochs) do |e|
  train_epoch(model, x_train, y_train, optimizer, batch_size: batch_size, seed: seed + e, epoch: e)
  evaluate(model, x_test, y_test, batch_size: 256)
end

