require "mkmf"
require "rubygems"

extension_name = "maurograd_ext"

# Find Numo NArray include path via gem specification

begin
  spec = Gem::Specification.find_by_name("numo-narray-alt")
rescue Gem::LoadError => e
  abort "numo-narray-alt not found, and its headers are needed to compile (#{e.class}: #{e.message}; gem search path: #{Gem.path.join(File::PATH_SEPARATOR)})"
end

numo_narray_candidates = Dir[File.join(spec.full_gem_path, "**", "numo", "narray.h")]

hdr =
  numo_narray_candidates.find { |p| p.include?("/ext/") } ||
  numo_narray_candidates.first

abort "Could not locate numo/narray.h inside #{spec.full_gem_path}" unless hdr

numo_inc = File.dirname(hdr, 2)
$INCFLAGS << " -I#{numo_inc}"

have_header("numo/narray.h") or abort "numo/narray.h not found even after adding include path"

# Choose the BLAS implementation.
#
# MAUROGRAD_BLAS=accelerate|openblas selects it explicitly. By default we use
# Apple's Accelerate framework on macOS and OpenBLAS everywhere else.
#
# Why Accelerate on macOS: on Apple Silicon its sgemm runs on the CPU's
# matrix units, and it is dramatically faster on the GEMM shapes
# convolutions produce. The weight gradient of a conv layer is a
# "tiny output, huge reduction" product ([Cout x rows] * [rows x K], with
# rows = N*OH*OW in the tens of thousands): for LeNet's first layer on a
# batch of 64, OpenBLAS (single thread) took 1.28 ms and Accelerate 0.06 ms.
default_blas = RUBY_PLATFORM.include?("darwin") ? "accelerate" : "openblas"
blas = (ENV["MAUROGRAD_BLAS"] || default_blas).downcase
abort "MAUROGRAD_BLAS must be 'accelerate' or 'openblas', got '#{blas}'" unless %w[accelerate openblas].include?(blas)
abort "MAUROGRAD_BLAS=accelerate is only available on macOS" if blas == "accelerate" && !RUBY_PLATFORM.include?("darwin")

# Flags (keep conservative; you can add -O3 -march=native later)
$CFLAGS << " -O3"
$CFLAGS << " -std=c11"

if blas == "accelerate"
  # ACCELERATE_NEW_LAPACK selects Accelerate's current CBLAS/LAPACK headers
  # (the old ones are deprecated since macOS 13.3).
  $CFLAGS << " -DMAUROGRAD_USE_ACCELERATE -DACCELERATE_NEW_LAPACK"
  $LDFLAGS << " -framework Accelerate"
  have_header("Accelerate/Accelerate.h") or abort "Accelerate framework headers not found (are the Xcode command line tools installed?)"
else
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

  # Check headers + library
  have_header("cblas.h") or abort "cblas.h not found (install openblas and/or set OPENBLAS_DIR)"
  have_library("openblas") or abort "libopenblas not found"
end

warn "[extconf] BLAS backend: #{blas}"

create_makefile(extension_name)
