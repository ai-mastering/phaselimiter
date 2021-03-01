#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "boost/filesystem.hpp"
#include "gflags/gflags.h"
#include "picojson.h"
#include <Eigen/Dense>

#include "bakuage/utils.h"
#include "bakuage/sound_quality.h"

DECLARE_string(analysis_data_dir);

void TestSoundQuality() {
    using boost::filesystem::recursive_directory_iterator;
    
    bakuage::SoundQualityCalculator calculator;
    
    recursive_directory_iterator last;
    for (recursive_directory_iterator itr(FLAGS_analysis_data_dir); itr != last; ++itr) {
        const std::string path = itr->path().string();
        if (!bakuage::StrEndsWith(path, ".json")) continue;
        
        std::ifstream ifs(path.c_str());
        std::string json_str((std::istreambuf_iterator<char>(ifs)),
                             std::istreambuf_iterator<char>());
        Eigen::VectorXd mean;
        Eigen::MatrixXd covariance;
        bakuage::SoundQualityCalculator::ParseReference(json_str.c_str(), &mean, &covariance);
        calculator.AddReference(json_str.c_str());
    }
    
    calculator.Prepare();
    
    for (recursive_directory_iterator itr(FLAGS_analysis_data_dir); itr != last; ++itr) {
        const std::string path = itr->path().string();
        if (!bakuage::StrEndsWith(path, ".json")) continue;
        
        std::ifstream ifs(path.c_str());
        std::string json_str((std::istreambuf_iterator<char>(ifs)),
                             std::istreambuf_iterator<char>());
        Eigen::VectorXd mean;
        Eigen::MatrixXd covariance;
        bakuage::SoundQualityCalculator::ParseReference(json_str.c_str(), &mean, &covariance);
        double sound_quality, lof;
        calculator.CalculateSoundQuality(mean, covariance, &sound_quality, &lof);
        std::cerr << path << " " << sound_quality << " " << lof << std::endl;
    }
    
    std::cerr << "border_lof " << calculator.border_lof() << std::endl;
    
}






