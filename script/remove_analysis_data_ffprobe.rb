# frozen_string_literal: true

require 'json'

dir = ARGV[0]

Dir.glob("#{dir}/**/*.json") do |path|
  warn path
  parsed = JSON.parse(File.read(path))
  parsed.delete('ffprobe')
  File.write(path, JSON.pretty_generate(parsed))
end
