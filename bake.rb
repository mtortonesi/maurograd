# Project-level bake tasks (https://bake.ioquatix.com/).
#
# `bake test` is provided by the bake-test gem and auto-detects sus - no
# task needed here for that. This file only adds what's specific to this
# project: building the native extension, and a `ci` task that does both
# in the order a fresh checkout actually needs.

ROOT = File.expand_path(__dir__)
BUILD_EXT = File.join(ROOT, "bin", "build_ext")

# Build the maurograd_ext native extension (extconf.rb + make).
def build_ext
  run_build_ext("build")
end

# Remove native extension build artifacts.
def clean_ext
  run_build_ext("clean")
end

# Clean and rebuild the native extension from scratch.
def rebuild_ext
  run_build_ext("rebuild")
end

# Build the extension, then run the full test suite - what CI runs.
def ci
  build_ext
  system("bundle", "exec", "sus", exception: true)
end

# Build the maurograd and maurograd-mlx gems into pkg/ (what gets pushed to
# rubygems.org: `gem push pkg/maurograd-<version>.gem`, and the same for
# maurograd-mlx). Gems are built from the files git tracks, so commit first.
def package
  require "fileutils"
  FileUtils.mkdir_p(File.join(ROOT, "pkg"))
  %w[maurograd maurograd-mlx].each do |name|
    Dir.chdir(ROOT) do
      system("gem", "build", "#{name}.gemspec", "--output", File.join("pkg", "#{name}-#{gem_version}.gem"), exception: true)
    end
  end
end

# Check code style with Standard (see .standard.yml for the couple of
# exceptions we actually need).
def lint
  system("bundle", "exec", "standardrb", exception: true)
end

# Auto-correct what Standard can fix safely.
def lint_fix
  system("bundle", "exec", "standardrb", "--fix", exception: true)
end

private

def gem_version
  require_relative "lib/maurograd/version"
  Maurograd::VERSION
end

def run_build_ext(subcommand)
  system(BUILD_EXT, subcommand, exception: true)
end
