# frozen_string_literal: true

require 'json'
require 'digest/md5'
require 'fileutils'
require 'pathname'

dir = ARGV[0]

Dir.glob("#{dir}/Music/**/*.json") do |path|
  relative_path = Pathname.new(path).relative_path_from(Pathname.new(dir)).to_s

  path2 = "#{dir}/#{Digest::MD5.hexdigest(relative_path)}.json"
  warn("#{relative_path} #{path} -> #{path2}")
  FileUtils.cp(path, path2)
end
