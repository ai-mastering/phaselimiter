#ifndef BAKUAGE_AUDIO_ANALYZER_STEREO_H_
#define BAKUAGE_AUDIO_ANALYZER_STEREO_H_

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <exception>
#include <memory>
#include "CImg.h"
#include "bakuage/utils.h"
#include "bakuage/pan_detect_filter.h"
#include "audio_analyzer/statistics.h"

namespace audio_analyzer {

template <typename Float>
void CalculateStereo(Float *input, int channels, int samples, int sample_freq, std::vector<std::vector<Float>> *_freq_pan_to_db) {  
    using namespace bakuage;

    if (channels != 2) {
        throw std::logic_error("CalculateStereo: channels must be 2");
    }

    std::vector<std::unique_ptr<bakuage::PanDetectFilter<double>>> filters;
    double freq = 50;
    while (freq < 20000) {
        double band_width = bakuage::GlasbergErb(freq);
        filters.push_back(std::unique_ptr<bakuage::PanDetectFilter<double>>(
            new bakuage::PanDetectFilter<double>(freq, band_width, sample_freq)));
        freq += band_width;
    }

	const int pan_division = 11; // 奇数
    std::vector<std::vector<Statistics>> freq_pan_to_energy(filters.size(), std::vector<Statistics>(pan_division));

    for (int i = 0; i < filters.size(); i++) {
        for (int j = 0; j < samples; j++) {
            bakuage::PanDetectFilter<double>::Output output = filters[i]->Clock(input[2 * j + 0], input[2 * j + 1]);           
            if (output.valid) {
                int pan_index = std::max<int>(0, std::min<int>(pan_division - 1, std::floor((0.5 + output.pan / 180) * pan_division)));
                for (int k = 0; k < pan_division; k++) {
                    double e = k == pan_index ? output.energy : 0;
                    freq_pan_to_energy[i][pan_index].Add(e);
                }
            }
        }
    }
    
    std::vector<std::vector<Float>> freq_pan_to_db(filters.size(), std::vector<Float>(pan_division));
    for (int i = 0; i < filters.size(); i++) {
        for (int j = 0; j < pan_division; j++) {
            freq_pan_to_db[i][j] = bakuage::ToDb(freq_pan_to_energy[i][j].mean());
        }
    }
    *_freq_pan_to_db = freq_pan_to_db;
}

struct StereoDistributionPoint {
	int x, y;
	float energy;
};

template <typename Float>
void WriteStereoDistributionPng(Float *input, int channels, int samples, int sample_freq, int image_width, int image_height, const char *output) {
	using namespace bakuage;
	using namespace cimg_library;
    
    typedef bakuage::PanDetectFilter<float> Filter;

	if (channels != 2) {
		throw std::logic_error("CalculateStereo: channels must be 2");
	}

	std::vector<std::unique_ptr<Filter>> filters;
	double freq = 50;
	while (freq < 20000) {
		double band_width = bakuage::GlasbergErb(freq);
		filters.push_back(std::unique_ptr<Filter>(new Filter(freq, band_width, sample_freq)));
		freq += band_width;
	}
    
    const double inv_180 = 1.0 / 180;
    const double inv_filter_size = 1.0 / filters.size();

	CImg<float> img(image_width, image_height, 1, 1, 0);
	for (int i = 0; i < filters.size(); i++) {		
		std::vector<StereoDistributionPoint> points;
		for (int j = 0; j < samples; j++) {
			const Filter::Output output = filters[i]->Clock(input[2 * j + 0], input[2 * j + 1]);

			if (output.valid) {
                
				StereoDistributionPoint point;
				point.x = std::max<int>(0, std::min<int>(image_width - 1, (image_width - 1) * (0.5 + output.pan * inv_180)));
				point.y = std::max<int>(0, std::min<int>(image_height - 1, (image_height - 1) * (1.0 - 1.0 * i * inv_filter_size)));
				point.energy = output.energy;
				points.push_back(point);
			}
		}
		const double scale = 1.0 * filters[i]->center_freq() / filters[i]->band_width()
			/ (1e-37 + points.size()); // 3dB/oct
		for (const auto &point : points) {
			img(point.x, point.y) += point.energy * scale;
		}
	}

	// x blur
	{
		const double rate = std::exp(-1.0 / (image_width * 0.03));
		for (int y = 0; y < image_height; y++) {
			double sum = 0;
			for (int x = 0; x < image_width; x++) {
				sum = sum * rate + img(x, y) * (1.0 - rate);
				img(x, y) = sum;
			}
			sum = 0;
			for (int x = image_width - 1; x >= 0; x--) {
				sum = sum * rate + img(x, y) * (1.0 - rate);
				img(x, y) = sum;
			}
		}
	}
	// y blur
	{
		const double rate = std::exp(-1.0 / (1.0 * image_height / filters.size()));
		for (int x = 0; x < image_width; x++) {
			double sum = 0;
			for (int y = 0; y < image_height; y++) {
				sum = sum * rate + img(x, y) * (1.0 - rate);
				img(x, y) = sum;
			}
			sum = 0;
			for (int y = image_height - 1; y >= 0; y--) {
				sum = sum * rate + img(x, y) * (1.0 - rate);
				img(x, y) = sum;
			}
		}
	}

	// normalize horizontal
	for (int y = 0; y < image_height; y++) {
		double max_value = 0;
		for (int x = 0; x < image_width; x++) {
			max_value = std::max<double>(max_value, img(x, y));
		}
		const double normalize_scale = 1.0 / (1e-37 + max_value);
		for (int x = 0; x < image_width; x++) {
			img(x, y) *= normalize_scale;
		}
	}

	// normalize and power
	double max_value = 0;
	for (int x = 0; x < image_width; x++) {
		for (int y = 0; y < image_height; y++) {
			max_value = std::max<double>(max_value, img(x, y));
		}
	}
	const double normalize_scale = 1.0 / (1e-37 + max_value);
	const double power = 0.4;

	// convert output image
	const float color[] = { 0.525f, 1.0f, 1.0f, 1.0f };
	CImg<unsigned char> output_img(image_width, image_height, 1, 3, 0);
	for (int i = 0; i < image_width; i++) {
		for (int j = 0; j < image_height; j++) {
			const double v = 1.6 * std::pow(img(i, j) * normalize_scale, power);
			for (int k = 0; k < 3; k++) {
				output_img(i, j, 0, k) = 255 * std::min<double>(1.0, color[k] * v);
			}
		}
	}
	output_img.save_png(output);
}

}

#endif 
