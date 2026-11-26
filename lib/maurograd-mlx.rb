# Entry point of the maurograd-mlx gem (what `require "maurograd-mlx"` and
# Bundler's automatic require load): Maurograd itself plus the MLX backend.
#
# Requiring it is optional: Maurograd loads the backend by itself the first
# time it is asked for (Maurograd.backend = :mlx, MAUROGRAD_BACKEND=mlx,
# tensor.to(:mlx)), as long as this gem is installed.
require "maurograd"
require "maurograd/backends/mlx"
