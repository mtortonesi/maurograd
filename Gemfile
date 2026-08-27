# frozen_string_literal: true

source 'https://rubygems.org'

# Core runtime + test suite dependencies (what CI installs).
gem 'sus', '~> 0.35.0'
gem 'numo-narray-alt', '~> 0.9.11'
gem 'numo-linalg-alt', '~> 0.7.1'

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
