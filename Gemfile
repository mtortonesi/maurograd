# frozen_string_literal: true

source 'https://rubygems.org'

# Runtime + development dependencies declared in the gemspec
# (numo-narray-alt, numo-linalg-alt, sus).
gemspec

# Task runner (see bake.rb). bake-test auto-detects and runs sus.
gem 'bake', '~> 0.25.0'
gem 'bake-test', '~> 0.3.0'

group :examples do
  gem 'chunky_png', '~> 1.4'
  gem 'erv', '~> 0.4.0'
  gem 'csv', '~> 3.3'
end

group :bench do
  gem 'torch-rb', '~> 0.23.1'
end

group :development do
  gem 'irb', '~> 1.15'
  gem 'debug', '~> 1.11'
  gem 'stackprof', '~> 0.2.27'
end
