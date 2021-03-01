#include "gflags/gflags.h"
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/filesystem.hpp>
#include "bakuage/sound_quality2.h"

DECLARE_string(analysis_data_dir);
DECLARE_string(sound_quality2_cache);
DECLARE_string(sound_quality2_cache_archiver);

void PrepareSoundQuality2() {
    using boost::filesystem::recursive_directory_iterator;
    
    bakuage::SoundQuality2Calculator calculator;
    recursive_directory_iterator last;
    std::vector<std::string> paths;
    for (recursive_directory_iterator itr(FLAGS_analysis_data_dir); itr != last; ++itr) {
        const std::string path = itr->path().string();
        if (!bakuage::StrEndsWith(path, ".json")) continue;
        paths.emplace_back(path);
    }
    calculator.PrepareFromPaths(paths.begin(), paths.end());
    if (FLAGS_sound_quality2_cache_archiver == "binary") {
		std::ofstream ofs(FLAGS_sound_quality2_cache, std::ios::binary);
        boost::archive::binary_oarchive oa(ofs);
        oa << calculator;
    } else if (FLAGS_sound_quality2_cache_archiver == "text") {
		std::ofstream ofs(FLAGS_sound_quality2_cache);
        boost::archive::text_oarchive oa(ofs);
        oa << calculator;
    } else {
        throw std::logic_error("unknown archive type " + FLAGS_sound_quality2_cache_archiver);
    }
}







