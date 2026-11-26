require "maurograd/bootstrap"

describe Maurograd::Bootstrap do
  it "defaults BLAS thread counts to 1 without overriding user choices" do
    vars = Maurograd::Bootstrap::BLAS_THREAD_VARS
    originals = vars.to_h { |v| [v, ENV[v]] }

    begin
      ENV.delete("OPENBLAS_NUM_THREADS")
      ENV["OMP_NUM_THREADS"] = "4"

      Maurograd::Bootstrap.limit_blas_threads!

      expect(ENV["OPENBLAS_NUM_THREADS"]).to be == "1"
      expect(ENV["OMP_NUM_THREADS"]).to be == "4"
    ensure
      originals.each { |v, val| val.nil? ? ENV.delete(v) : ENV[v] = val }
    end
  end
end
