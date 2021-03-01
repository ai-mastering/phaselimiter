# frozen_string_literal: true

require 'open3'
require 'shellwords'
require 'tmpdir'

def find_audio_stream(probe_result)
  probe_result['streams'].find { |stream| stream['codec_type'] == 'audio' }
end

def exec_command(command)
  stdout, status = Open3.capture2(command)
  raise "failed_command:#{command}\tstatus:#{status}" unless status.success?

  stdout
end

def ffprobe(path)
  JSON.parse(exec_command(Shellwords.join(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', path]).to_s))
end

def get_info(path)
  output = exec_command(Shellwords.join(['sndfile-info', path]).to_s)
  {
    frames: output.match(/Frames\s*:\s*(\d+)/)[1].to_i
  }
end

describe 'phase_limiter' do
  exec = lambda do |options|
    path = nil
    Dir.glob('bin/**/phase_limiter*') do |p|
      next if p.match?(/pdb$/)

      path = p
    end
    command = [
      path,
      '--mastering_reference_file=resource/mastering_reference.json',
      '--mastering_reverb_ir_left=test_data/test5.wav',
      '--mastering_reverb_ir_right=test_data/test5.wav',
      '--mastering_reverb_gain=0',
      '--mastering_reverb_predelay=0.04',
      '--mastering_reverb_drr_range=30',
      options
    ].join(' ')
    warn command
    o, e, s = Open3.capture3(command)
    raise "command failed #{o} #{e} #{s}" unless s == 0

    o
  end

  describe 'smoke test' do
    it 'do nothing' do
      Dir.mktmpdir do |dir|
        exec.call("--input=test_data/test2.wav --output=#{dir}/output.wav --reference_mode=peak --reference=0 --pre_compression=false")
      end
    end
    it 'simple limiting' do
      Dir.mktmpdir do |dir|
        exec.call("--input=test_data/test2.wav --output=#{dir}/output.wav")
      end
    end
    it 'simple limiting short wav' do
      Dir.mktmpdir do |dir|
        exec.call("--input=test_data/test5.wav --output=#{dir}/output.wav")
      end
    end
    it 'simple limiting output 24bit wav' do
      Dir.mktmpdir do |dir|
        exec.call("--input=test_data/test2.wav --output=#{dir}/output.wav --bit_depth=24")
      end
    end
    it 'simple limiting input mp3 output mp3' do
      Dir.mktmpdir do |dir|
        exec.call("--input=test_data/test1.mp3 --output=#{dir}/output.mp3 --output_format=mp3")
      end
    end
    it 'many workers' do
      Dir.mktmpdir do |dir|
        exec.call("--input=test_data/test2.wav --output=#{dir}/output.wav --worker_count=64")
      end
    end
    it 'with mastering' do
      Dir.mktmpdir do |dir|
        exec.call("--input=test_data/test2.wav --output=#{dir}/output.wav --mastering=true")
      end
    end
    it 'with mastering and reverb' do
      Dir.mktmpdir do |dir|
        exec.call("--input=test_data/test2.wav --output=#{dir}/output.wav --mastering=true --mastering_reverb=true")
      end
    end
    it 'range mastering' do
      Dir.mktmpdir do |dir|
        exec.call("--input=test_data/test2.wav --output=#{dir}/output.wav --mastering=true --start_at=0.7 --end_at=1.7")
      end
    end
    it 'mastering3' do
      Dir.mktmpdir do |dir|
        exec.call("--input=test_data/test2.wav --output=#{dir}/output.wav --mastering=true --mastering_mode=mastering3")
      end
    end
  end

  # 同じサンプリングレート、同じフォーマットならサンプルレベルで一致させる
  describe 'sync test' do
    [
      {
        input_format: :aac,
        output_format: :aac
      },
      {
        input_format: :aac,
        output_format: :wav
      },
      {
        input_format: :mp3,
        output_format: :mp3
      },
      {
        input_format: :mp3,
        output_format: :wav
      },
      {
        input_format: :wav,
        output_format: :wav
      }
    ].each do |test|
      [44_100].each do |sample_rate| # 48000は一旦目を瞑る
        [2000, 44_100, 56_789].each do |samples|
          describe "#{test[:input_format]} -> #{test[:output_format]} #{sample_rate}Hz #{samples}samples" do
            it 'match per sample' do
              Dir.mktmpdir do |dir|
                input_path = "#{dir}/input.#{test[:input_format]}"
                output_path = "#{dir}/output.#{test[:output_format]}"
                File.open("#{dir}/tmp.raw", 'wb') do |io|
                  io.write((1..samples).map { |i| i == 1 ? ((1 << 15) - 1) : 0 }.pack('s*'))
                end
                exec_command(Shellwords.join(['ffmpeg', '-f', 's16le', '-ar', sample_rate, '-ac', 1, '-i', "#{dir}/tmp.raw", input_path]).to_s)

                exec.call("--input=#{input_path} --output=#{output_path} --output_format=#{test[:output_format]} --sample_rate=#{sample_rate}")

                exec_command(Shellwords.join(['ffmpeg', '-i', input_path, "#{dir}/input_decoded.wav"]).to_s)
                exec_command(Shellwords.join(['ffmpeg', '-i', output_path, "#{dir}/output_decoded.wav"]).to_s)

                input_frames = get_info("#{dir}/input_decoded.wav")[:frames]
                output_frames = get_info("#{dir}/output_decoded.wav")[:frames]

                expect(output_frames).to eq(input_frames)
              end
            end
          end
        end
      end
    end
  end
end
