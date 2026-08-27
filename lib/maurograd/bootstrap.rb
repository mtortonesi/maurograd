module Maurograd
  module Bootstrap
    # Common install locations for libopenblas on Linux distros that don't
    # expose it via ldconfig with an easily-parsed name (e.g. minimal
    # containers without ldconfig's cache populated yet).
    LINUX_CANDIDATE_DIRS = [
      "/usr/lib/x86_64-linux-gnu",
      "/usr/lib/aarch64-linux-gnu",
      "/usr/local/lib",
      "/usr/lib",
    ].freeze

    def self.ensure_openblas!
      # Don't override if the user has already set it.
      ENV["NUMO_LINALG_BACKEND"] ||= "openblas"

      return unless ENV["OPENBLAS_LIBPATH"].nil?

      libpath = detect_via_homebrew || detect_via_ldconfig || detect_via_known_dirs
      ENV["OPENBLAS_LIBPATH"] = libpath if libpath
      # If nothing was found, stay silent: numo-linalg will try to
      # auto-detect the backend on its own.
    end

    def self.detect_via_homebrew
      brew = `brew --prefix openblas`.strip
      brew.empty? ? nil : File.join(brew, "lib")
    rescue StandardError
      nil
    end

    def self.detect_via_ldconfig
      output = `ldconfig -p`
      line = output.each_line.find { |l| l.include?("libopenblas") }
      libpath_from_ldconfig_line(line)
    rescue StandardError
      nil
    end

    # Extracted for testability: parses one line of `ldconfig -p` output,
    # e.g. "\tlibopenblas.so.0 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libopenblas.so.0"
    def self.libpath_from_ldconfig_line(line)
      return nil unless line

      target = line.split("=>").last
      return nil unless target

      File.dirname(target.strip)
    end

    def self.detect_via_known_dirs
      LINUX_CANDIDATE_DIRS.find do |dir|
        Dir.exist?(dir) && !Dir.glob(File.join(dir, "libopenblas*")).empty?
      end
    end
  end
end
