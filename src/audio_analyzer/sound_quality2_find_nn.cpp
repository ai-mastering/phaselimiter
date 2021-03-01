#include "gflags/gflags.h"
#include <iostream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/filesystem.hpp>
#include "tbb/tbb.h"
#include "tbb/concurrent_vector.h"
#include "bakuage/sound_quality2.h"

DECLARE_string(input);
DECLARE_string(analysis_data_dir);
DECLARE_string(sound_quality2_cache);
DECLARE_string(sound_quality2_cache_archiver);

namespace {
    struct Item {
        std::string path;
        double distance;
    };
}

// 学習済みのcacheを使って距離を計算する (前処理などを利用するために、学習済みのcacheが必要)
void SoundQuality2FindNn() {
    using boost::filesystem::recursive_directory_iterator;
    
    bakuage::SoundQuality2Calculator calculator;
    {
        if (FLAGS_sound_quality2_cache_archiver == "binary") {
			std::ifstream ifs(FLAGS_sound_quality2_cache, std::ios::binary);
            boost::archive::binary_iarchive ia(ifs);
            ia >> calculator;
        } else if (FLAGS_sound_quality2_cache_archiver == "text") {
			std::ifstream ifs(FLAGS_sound_quality2_cache);
            boost::archive::text_iarchive ia(ifs);
            ia >> calculator;
        } else {
            throw std::logic_error("unknown archive type " + FLAGS_sound_quality2_cache_archiver);
        }
    }
    
    Eigen::VectorXd mean;
    Eigen::MatrixXd cov;
    bakuage::SoundQuality2CalculatorUnit::ParseReference(bakuage::LoadStrFromFile(FLAGS_input.c_str()).c_str(), &mean, &cov);
    bakuage::MasteringReference2 target(mean, cov);
    
    tbb::concurrent_vector<Item> items;
    
    recursive_directory_iterator last;
    std::vector<std::string> paths;
    for (recursive_directory_iterator itr(FLAGS_analysis_data_dir); itr != last; ++itr) {
        const std::string path = itr->path().string();
        if (!bakuage::StrEndsWith(path, ".json")) continue;
        paths.emplace_back(path);
    }
    
    tbb::parallel_for<int>(0, paths.size(), [&paths, &items, &calculator, &target](int i) {
        Eigen::VectorXd mean;
        Eigen::MatrixXd cov;
        bakuage::SoundQuality2CalculatorUnit::ParseReference(bakuage::LoadStrFromFile(paths[i].c_str()).c_str(), &mean, &cov);
        bakuage::MasteringReference2 reference(mean, cov);
        Item item;
        item.path = paths[i];
        item.distance = calculator.CalculateDistance(reference, target);
        items.emplace_back(item);
    });
    
    std::sort(items.begin(), items.end(), [](const Item &a, const Item &b) {
        return a.distance < b.distance;
    });
    
    for (int i = 0; i < std::min<int>(100, items.size()); i++) {
        std::cout << "distance:" << items[i].distance << "\tpath:" << items[i].path << std::endl;
    }
}
