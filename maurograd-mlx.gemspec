# frozen_string_literal: true

require_relative "lib/maurograd/version"

# The MLX backend for Maurograd, as its own gem: it needs mlx-c (a Homebrew
# library, Apple Silicon only), so keeping it out of the main gem means a
# missing or incompatible mlx-c can never break installing Maurograd itself.
#
# Released from the same repository and at the same version as maurograd.
Gem::Specification.new do |spec|
  spec.name = "maurograd-mlx"
  spec.version = Maurograd::VERSION
  spec.authors = ["Mauro Tortonesi"]
  spec.email = ["mauro.tortonesi@unife.it"]

  spec.summary = "Apple Silicon GPU backend for Maurograd, through MLX."
  spec.description = <<~DESC
    Adds an :mlx backend to Maurograd that runs tensors and training on the
    Apple Silicon GPU, through Ruby bindings for mlx-c (the C API of Apple's
    MLX array framework). Requires mlx-c: `brew install mlx-c` before
    installing this gem. Select it with MAUROGRAD_BACKEND=mlx,
    Maurograd.backend = :mlx, or model.to(:mlx).
  DESC
  spec.homepage = "https://github.com/mtortonesi/maurograd"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.2"

  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["bug_tracker_uri"] = "#{spec.homepage}/issues"
  spec.metadata["changelog_uri"] = "#{spec.homepage}/blob/main/CHANGELOG.md"
  spec.metadata["rubygems_mfa_required"] = "true"

  spec.files = Dir.chdir(__dir__) do
    `git ls-files -z`.split("\x0").select do |f|
      f.start_with?("ext/maurograd_mlx/") ||
        %w[lib/maurograd/backends/mlx.rb lib/maurograd-mlx.rb LICENSE].include?(f)
    end
  end

  spec.require_paths = ["lib"]
  spec.extensions = ["ext/maurograd_mlx/extconf.rb"]

  # Same minor version as maurograd: the backend interface is internal and
  # may change between minor releases before 1.0.
  major, minor, = Maurograd::VERSION.split(".")
  spec.add_dependency "maurograd", "~> #{major}.#{minor}.0"
  # Its headers are needed to compile the extension.
  spec.add_dependency "numo-narray-alt", "~> 0.9.11"
end
