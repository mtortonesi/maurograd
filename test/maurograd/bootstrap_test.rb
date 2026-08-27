require 'maurograd/bootstrap'

describe Maurograd::Bootstrap do
  it "parses the target path out of a normal ldconfig -p line" do
    line = "\tlibopenblas.so.0 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libopenblas.so.0\n"
    expect(Maurograd::Bootstrap.libpath_from_ldconfig_line(line)).to be == "/usr/lib/x86_64-linux-gnu"
  end

  it "parses a line without an architecture annotation" do
    line = "\tlibopenblas.so (libc6) => /usr/lib/libopenblas.so\n"
    expect(Maurograd::Bootstrap.libpath_from_ldconfig_line(line)).to be == "/usr/lib"
  end

  it "returns nil for a nil line (no match found upstream)" do
    expect(Maurograd::Bootstrap.libpath_from_ldconfig_line(nil)).to be == nil
  end

  it "sets NUMO_LINALG_BACKEND without overriding an existing value" do
    original_backend = ENV["NUMO_LINALG_BACKEND"]
    original_libpath = ENV["OPENBLAS_LIBPATH"]

    begin
      ENV["NUMO_LINALG_BACKEND"] = "custom"
      ENV.delete("OPENBLAS_LIBPATH")

      Maurograd::Bootstrap.ensure_openblas!

      expect(ENV["NUMO_LINALG_BACKEND"]).to be == "custom"
    ensure
      original_backend.nil? ? ENV.delete("NUMO_LINALG_BACKEND") : ENV["NUMO_LINALG_BACKEND"] = original_backend
      original_libpath.nil? ? ENV.delete("OPENBLAS_LIBPATH") : ENV["OPENBLAS_LIBPATH"] = original_libpath
    end
  end

  it "does not override an already-set OPENBLAS_LIBPATH" do
    original_libpath = ENV["OPENBLAS_LIBPATH"]

    begin
      ENV["OPENBLAS_LIBPATH"] = "/already/set"

      Maurograd::Bootstrap.ensure_openblas!

      expect(ENV["OPENBLAS_LIBPATH"]).to be == "/already/set"
    ensure
      original_libpath.nil? ? ENV.delete("OPENBLAS_LIBPATH") : ENV["OPENBLAS_LIBPATH"] = original_libpath
    end
  end
end
