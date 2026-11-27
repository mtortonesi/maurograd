# frozen_string_literal: true

require_relative "lib/maurograd/version"

Gem::Specification.new do |spec|
  spec.name = "maurograd"
  spec.version = Maurograd::VERSION
  spec.authors = ["Mauro Tortonesi"]
  spec.email = ["mauro.tortonesi@unife.it"]

  spec.summary = "A didactic tensor autograd framework for Ruby, with CNN support."
  spec.description = <<~DESC
    Maurograd is a tensor-based automatic differentiation framework for
    Ruby, written to be read end to end. Tensors and the autograd engine are
    plain Ruby; the math runs on a pluggable backend. The default CPU
    backend uses Numo::NArray plus a C extension for convolution, pooling,
    ReLU and BatchNorm (BLAS from Apple's Accelerate on macOS, OpenBLAS
    elsewhere). Includes SGD with momentum, Adam, Dropout and BatchNorm.
    An Apple Silicon GPU backend is available separately, in the
    maurograd-mlx gem.
  DESC
  spec.homepage = "https://github.com/mtortonesi/maurograd"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.3"

  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["bug_tracker_uri"] = "#{spec.homepage}/issues"
  spec.metadata["changelog_uri"] = "#{spec.homepage}/blob/main/CHANGELOG.md"
  spec.metadata["rubygems_mfa_required"] = "true"

  # An explicit allowlist, so development files (Gemfile, bake.rb, test/,
  # bench/, config/, ...) never end up in the gem. The MLX backend and its
  # extension ship in maurograd-mlx (see maurograd-mlx.gemspec).
  spec.files = Dir.chdir(__dir__) do
    `git ls-files -z`.split("\x0").select do |f|
      next false if f == "lib/maurograd/backends/mlx.rb" || f == "lib/maurograd-mlx.rb"

      f.start_with?("lib/", "ext/maurograd_ext/", "examples/") ||
        %w[README.md LICENSE CHANGELOG.md].include?(f)
    end
  end

  spec.require_paths = ["lib"]
  spec.extensions = ["ext/maurograd_ext/extconf.rb"]

  spec.add_dependency "numo-narray-alt", "~> 0.9.11"
  spec.add_dependency "numo-linalg-alt", "~> 0.7.1"

  spec.add_development_dependency "sus", "~> 0.35.0"
end
