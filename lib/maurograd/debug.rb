module Maurograd
  # Off by default: expensive checks like Utils.assert_finite! are pure
  # overhead once a model is known to be numerically stable, since they
  # walk the entire array on every forward/backward call.
  #
  # Enable with `Maurograd.debug = true`, or by setting MAUROGRAD_DEBUG=1
  # in the environment before requiring maurograd.
  @debug = ENV['MAUROGRAD_DEBUG'] == '1'

  class << self
    attr_accessor :debug
  end
end
