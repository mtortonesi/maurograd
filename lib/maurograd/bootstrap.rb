module Maurograd
  # Process-level setup that has to happen before the native libraries load.
  module Bootstrap
    # Environment variables that cap the thread pools of the BLAS/OpenMP
    # libraries we load.
    BLAS_THREAD_VARS = %w[OPENBLAS_NUM_THREADS OMP_NUM_THREADS].freeze

    # Defaults BLAS/OpenMP thread pools to a single thread, unless the user
    # has already chosen a value.
    #
    # Why: numo-linalg-alt (used by Linear/MatMul via Numo's #dot) bundles
    # its own OpenBLAS, whose worker threads busy-wait between calls and
    # compete with everything else for the cores. Where maurograd_ext also
    # links OpenBLAS (Linux), that is two such pools starving each other: on
    # a 10-core Apple M5 built against OpenBLAS, one LeNet training step went
    # from 13 ms (single-threaded) to 278 ms (default threading). With the
    # extension on Accelerate (macOS) it is milder but still a loss: 4.9 ms
    # vs 9.3 ms. At the matrix sizes Maurograd works with, multithreaded
    # BLAS has little to gain anyway.
    #
    # This must run before those libraries are loaded: both read the
    # variables once, when they initialize. To experiment with threading,
    # set the variables explicitly in the environment of the process.
    def self.limit_blas_threads!
      BLAS_THREAD_VARS.each { |var| ENV[var] ||= "1" }
    end
  end
end
