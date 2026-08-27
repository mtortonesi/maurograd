require "mkmf"
require "rubygems"

extension_name = "maurograd_ext"

# Find Numo NArray include path via gem specification

begin
  spec = Gem::Specification.find_by_name("numo-narray-alt")
rescue Gem::LoadError
  abort "numo-narray-alt gem not found (is it in your Gemfile and bundle install done?)"
end

numo_narray_candidates = Dir[File.join(spec.full_gem_path, "**", "numo", "narray.h")]

hdr =
  numo_narray_candidates.find { |p| p.include?("/ext/") } ||
  numo_narray_candidates.first

abort "Could not locate numo/narray.h inside #{spec.full_gem_path}" unless hdr

numo_inc = File.dirname(File.dirname(hdr))
$INCFLAGS << " -I#{numo_inc}"

dir_config("numo", numo_inc)

have_header("numo/narray.h") or abort "numo/narray.h not found even after adding include path"

# Find OpenBLAS include and lib paths

# Try pkg-config first (portable across distros; how apt's libopenblas-dev
# and most Linux package managers register themselves).
pkg_config("openblas")

# Prefer Homebrew OpenBLAS if present; otherwise fall back to common
# Linux system prefixes (apt's libopenblas-dev installs under plain /usr).
openblas_candidates = [
  ENV["OPENBLAS_DIR"],
  "/opt/homebrew/opt/openblas",   # Apple Silicon
  "/usr/local/opt/openblas",      # Intel Homebrew
  "/usr"                          # common Linux system install
].compact.uniq

openblas_inc = nil
openblas_lib = nil

openblas_candidates.each do |prefix|
  inc = File.join(prefix, "include")
  lib = File.join(prefix, "lib")
  next unless File.directory?(inc) && File.directory?(lib)
  openblas_inc = inc
  openblas_lib = lib
  break
end

# Adds -I and -L only if found
if openblas_inc && openblas_lib
  dir_config("openblas", openblas_inc, openblas_lib)
end

# Flags (keep conservative; you can add -O3 -march=native later)
$CFLAGS << " -O3"
$CFLAGS << " -std=c11"

# Check headers + library
have_header("cblas.h") or abort "cblas.h not found (install openblas and/or set OPENBLAS_DIR)"
have_library("openblas") or abort "libopenblas not found"

create_makefile(extension_name)
