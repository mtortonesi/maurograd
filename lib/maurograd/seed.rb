module Maurograd
  # Numo::NArray seeds its own global RNG to the fixed constant 0 at load
  # time (see numo-narray's ext/numo/narray/rand.c, Init_nary_rand), not
  # from OS entropy. Left alone, that makes any unseeded Numo.rand/rand_norm
  # call - including Layers::Conv2D/Linear's weight init when no seed: is
  # given - silently identical across every process run. Re-randomize it
  # here at load time so "no seed given" genuinely means random; the
  # inverse case (asking for a reproducible run) is exactly what
  # Maurograd.seed! is for.
  Numo::NArray.srand

  # Seeds every source of randomness Maurograd's layers rely on by default,
  # so a full run becomes reproducible from one call - instead of having to
  # pass seed: to every single layer by hand.
  #
  # Affects:
  # - Weight initialization in any layer constructed without its own
  #   explicit seed: (Layers::Conv2D, Layers::Linear) - they fall back to
  #   Numo::NArray's global RNG, which this reseeds.
  # - Ruby's own Kernel#rand, in case other code relies on it.
  #
  # Does NOT affect Datasets::Batching.batches, which already has its own
  # independent seed: (default 0, so it's already deterministic on its own)
  # - pass seed: there directly if you also want to control data order.
  def self.seed!(n)
    Numo::NArray.srand(n)
    srand(n)
    n
  end
end
