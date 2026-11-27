require "mkmf"
require "rubygems"

# Optional extension: Ruby bindings for mlx-c, used by the MLX backend
# (lib/maurograd/backends/mlx.rb). Only buildable where mlx-c is installed
# (macOS: `brew install mlx-c`); bin/build_ext skips it otherwise.

extension_name = "maurograd_mlx"

# Numo NArray headers (same lookup as maurograd_ext).
begin
  spec = Gem::Specification.find_by_name("numo-narray-alt")
rescue Gem::LoadError => e
  abort "numo-narray-alt not found, and its headers are needed to compile (#{e.class}: #{e.message}; gem search path: #{Gem.path.join(File::PATH_SEPARATOR)})"
end
hdr = Dir[File.join(spec.full_gem_path, "**", "numo", "narray.h")].then do |c|
  c.find { |p| p.include?("/ext/") } || c.first
end
abort "Could not locate numo/narray.h inside #{spec.full_gem_path}" unless hdr
$INCFLAGS << " -I#{File.dirname(hdr, 2)}"
have_header("numo/narray.h") or abort "numo/narray.h not found even after adding include path"

# mlx-c: MLX_C_DIR=<prefix> selects it explicitly; otherwise ask Homebrew.
mlx_c_dir = ENV["MLX_C_DIR"]
if mlx_c_dir.nil?
  brew = begin
    `brew --prefix mlx-c 2>/dev/null`.strip
  rescue
    ""
  end
  mlx_c_dir = brew unless brew.empty?
end
abort "mlx-c not found (install it with `brew install mlx-c`, or set MLX_C_DIR)" unless mlx_c_dir && File.directory?(mlx_c_dir)

mlx_inc = File.join(mlx_c_dir, "include")
mlx_lib = File.join(mlx_c_dir, "lib")
$INCFLAGS << " -I#{mlx_inc}"
$LDFLAGS << " -L#{mlx_lib} -Wl,-rpath,#{mlx_lib}"
$CFLAGS << " -O2 -std=c11"

have_header("mlx/c/mlx.h") or abort "mlx/c/mlx.h not found under #{mlx_inc}"
have_library("mlxc") or abort "libmlxc not found under #{mlx_lib}"

create_makefile(extension_name)
