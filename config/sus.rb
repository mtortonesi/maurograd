# Add lib folder to the Ruby LOAD_PATH
base_directory = ::File.expand_path("..", __dir__)
$LOAD_PATH.unshift(::File.join(base_directory, "lib"))

# Load main file of the framework
require "maurograd"
