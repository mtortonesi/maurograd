# frozen_string_literal: true

# Loads the native extension if present.
# In dev mode (repo checkout), the compiled artifact typically lives under ext/**/.
# In installed gem mode, it may be directly require-able as "maurograd_ext".

begin
  require "maurograd_ext"
rescue LoadError
  project_root = File.expand_path("../..", __dir__) # .../maurograd
  pattern = File.join(project_root, "ext", "**", "maurograd_ext.{bundle,so}")

  candidates = Dir.glob(pattern).sort
  if candidates.empty?
    raise LoadError, <<~MSG
      cannot load native extension 'maurograd_ext'
      Looked for: #{pattern}

      Build it with:
        ./bin/build_ext

      (If you're on macOS: expect a .bundle; on Linux: a .so)
    MSG
  end

  # Requiring the absolute path is fine (Ruby will load the .bundle/.so).
  require candidates.first
end

# Sanity check (optional but helps catch partial loads early)
unless defined?(Maurograd::Ext) && 
    Maurograd::Ext.respond_to?(:conv2d_forward) && 
    Maurograd::Ext.respond_to?(:im2col) && 
    Maurograd::Ext.respond_to?(:col2im)
  raise LoadError, 'maurograd_ext loaded but key methods not defined'
end

