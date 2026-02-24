module Maurograd
  module Bootstrap
    def self.ensure_openblas!
      # Don't override if the user has already set it.
      ENV["NUMO_LINALG_BACKEND"] ||= "openblas"

      if ENV["OPENBLAS_LIBPATH"].nil?
        begin
          brew = `brew --prefix openblas`.strip
          if !brew.empty?
            ENV["OPENBLAS_LIBPATH"] = File.join(brew, "lib")
          end
        rescue StandardError
          # Silent fallback: numo-linalg will auto-detect the backend.
        end
      end
    end
  end
end

