# frozen_string_literal: true

require_relative 'lib/maurograd/version'

Gem::Specification.new do |spec|
  spec.name        = 'maurograd'
  spec.version      = Maurograd::VERSION
  spec.authors      = ['Mauro Tortonesi']
  spec.email        = ['mauro.tortonesi@unife.it']

  spec.summary      = 'A didactic tensor autograd framework for Ruby, with CNN support.'
  spec.description  = <<~DESC
    Maurograd is a tensor-based automatic differentiation framework for
    Ruby, built on Numo::NArray, with 2D convolution implemented in a C
    extension (im2col + GEMM + col2im) backed by OpenBLAS. It started as a
    clone of Andrej Karpathy's micrograd and grew into something more
    ambitious: a reasonably CPU-competent framework for building and
    training small CNNs, designed to be simple enough to read end to end.
  DESC
  spec.homepage     = 'https://github.com/mtortonesi/maurograd'
  spec.license      = 'MIT'
  spec.required_ruby_version = '>= 3.0'

  spec.metadata['source_code_uri'] = spec.homepage
  spec.metadata['bug_tracker_uri'] = "#{spec.homepage}/issues"

  spec.files = Dir.chdir(__dir__) do
    `git ls-files -z`.split("\x0").reject do |f|
      f.match(%r{\A(?:test|bench|\.github)/}) || f == '.gitignore'
    end
  end

  spec.require_paths = ['lib']
  spec.extensions     = ['ext/maurograd_ext/extconf.rb']

  spec.add_dependency 'numo-narray-alt', '~> 0.9.11'
  spec.add_dependency 'numo-linalg-alt', '~> 0.7.1'

  spec.add_development_dependency 'sus', '~> 0.35.0'
end
