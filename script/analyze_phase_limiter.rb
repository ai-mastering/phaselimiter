# frozen_string_literal: true

require 'json'
require 'thor'

class MyCLI < Thor
  desc 'eval [options]', 'calc eval func changing parameters'
  option :erb_eval_func_weighting, type: :array, default: [0, 1], desc: 'array of erb_eval_func_weighting option(0/)1'
  option :input, type: :array, desc: 'array of input file'
  option :loudness, type: :array, default: [-12, -9, -7, -5, -3], desc: 'array of target loudness count. Thorの仕様でマイナスを指定できないので、マイナスは-1はm1と表現する'
  option :iter, type: :array, default: [25, 50, 75, 100, 150, 200, 300, 400], desc: 'array of iter count'
  option :initial_noise, type: :array, default: [0.25, 0.5, 1, 2, 4, 8], desc: 'array of noise update initial noise count'
  def eval
    options[:erb_eval_func_weighting].each do |erb_eval_func_weighting|
      options[:input].each do |input|
        options[:loudness].each do |loudness_str|
          loudness = loudness_str[0] == 'm' ? -loudness_str[1..-1].to_f : loudness_str.to_f
          options[:iter].each do |iter|
            options[:initial_noise].each do |initial_noise|
              command = [
                '(cd resource &&',
                '../bin/Release/phase_limiter',
                # '../phase_limiter2',
                "--input=../#{input}",
                "'--output=/tmp/output_#{input.gsub(%r{.*/}, '').gsub(/\..*/, '')}_#{loudness}_#{iter}_#{erb_eval_func_weighting}_#{initial_noise}.wav'",
                "--reference=#{loudness}",
                "--max_iter1=#{iter}",
                "--max_iter2=#{10 * iter.to_i}",
                "-erb_eval_func_weighting=#{erb_eval_func_weighting.to_i != 0 ? 'true' : 'false'}",
                # "--noise_update_fista_enable_ratio=0.5",
                "--noise_update_initial_noise=#{initial_noise}",
                '2>&1)'
              ].join(' ')
              # https://techracho.bpsinc.jp/hachi8833/2018_06_01/56612
              started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
              out = `#{command}`
              finished_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
              last_normalized_eval = out.scan(/normalized_eval:([^\s]+)/)&.last&.[](0)&.to_f
              unless last_normalized_eval
                warn "error:#{out}"
                next
              end
              puts "erb_eval_func_weighting:#{erb_eval_func_weighting}\tinput:#{input}\tloudness:#{loudness}\titer:#{iter}\tinitial_noise:#{initial_noise}\tnormalized_eval:#{last_normalized_eval}\telapsed_sec:#{finished_at - started_at}"
            end
          end
        end
      end
    end
  end
end

MyCLI.start(ARGV)
