# frozen_string_literal: true

require 'json'
require 'digest/md5'
require 'fileutils'
require 'pathname'

path = ARGV[0]
root_dir = './resource/analysis_data'

parsed = JSON.parse(File.read(path))
paths = parsed['paths']

(0..(paths.length - 1)).each do |i|
  relative_path = Pathname.new(paths[i]).relative_path_from(Pathname.new(root_dir)).to_s
  path2 = "#{root_dir}/#{Digest::MD5.hexdigest(relative_path)}.json"

  raise "not found #{relative_path} #{path2}" unless File.exist?(path2)

  warn("#{paths[i]} -> #{path2}")
  paths[i] = path2
end

File.write(path, JSON.pretty_generate(parsed))
