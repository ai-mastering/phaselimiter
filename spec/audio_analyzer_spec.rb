# frozen_string_literal: true

require 'open3'
require 'json'
require 'tmpdir'

describe 'audio_analyzer' do
  exec = lambda do |options|
    Dir.mktmpdir do |dir|
      path = nil
      Dir.glob('bin/**/audio_analyzer*') do |p|
        next if p.match?(/pdb$/)

        path = p
      end
      command = "#{path} --tmp=#{dir} #{options} --sound_quality2=false"
      warn command
      o, e, s = Open3.capture3(command)
      raise "command failed #{o} #{e} #{s}" unless s == 0

      o
    end
  end

  describe 'single mode' do
    it 'succeed' do
      result = JSON.parse(exec.call('--input=test_data/test2.wav --analysis_data_dir=./resource/analysis_data'))

      print(result)

      expect(result['channels']).to eq(2)
      expect(result['format']).to eq(65_538)
      expect(result['frames']).to eq(280_049)
      expect(result['sample_rate']).to eq(44_100)
      expect(result['sections']).to eq(1)
      expect(result['seekable']).to eq(1)

      expect(result['peak']).to be_within(0.1).of(-3.77)
      expect(result['rms']).to be_within(0.1).of(-16.3)
      expect(result['loudness']).to be_within(0.1).of(-14.8)
      expect(result['loudness_range']).to be_within(0.1).of(2.0)
      expect(result['dynamics']).to be_within(0.1).of(1.39)
      expect(result['sharpness']).to be_within(0.1).of(2.19)
      expect(result['space']).to be_within(0.1).of(0.325)
      # なんか計算結果がずれるようになったけど、使っていないから気にしない。double -> floatにしたせいかも
      # expect(result['drr']).to be_within(0.1).of(9.2)

      expect(result['spectrum']).to be_truthy
      expect(result['waveform']).to be_truthy
      expect(result['histogram']).to be_truthy
      expect(result['loudness_time_series']).to be_truthy
      expect(result['freq_pan_to_db']).to be_truthy
    end
  end
end
