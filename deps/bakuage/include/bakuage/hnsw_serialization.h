#ifndef bakuage_hnsw_serialization_h
#define bakuage_hnsw_serialization_h

#include <fstream>
#include <boost/filesystem.hpp>
#include "hnswlib/hnswlib.h"
#include "bakuage/memory.h"
#include "bakuage/utils.h"

namespace bakuage {
    template<class Archive, class T>
    void SaveHnswToArchive(Archive & ar, const hnswlib::HierarchicalNSW<T> &g, const unsigned int version) {
        const auto temp_path = NormalizeToString((boost::filesystem::temp_directory_path() / boost::filesystem::unique_path()).native());
        const_cast<hnswlib::HierarchicalNSW<T> &>(g).saveIndex(temp_path); // 実質constだがconstがついていない
        bakuage::AlignedPodVector<char> buffer;
        {
            std::ifstream ifs(temp_path, std::ios::binary | std::ios::ate);
            std::streamsize size = ifs.tellg();
            ifs.seekg(0, std::ios::beg);
            buffer.resize(size);
            ifs.read(buffer.data(), size);
        }
        boost::filesystem::remove(temp_path);
        ar << buffer;
    }
    
    template<class Archive, class T>
    void LoadHnswFromArchive(Archive & ar, hnswlib::HierarchicalNSW<T> &g, const unsigned int version, hnswlib::SpaceInterface<T> *space) {
        const auto temp_path = NormalizeToString((boost::filesystem::temp_directory_path() / boost::filesystem::unique_path()).native());
        bakuage::AlignedPodVector<char> buffer;
        ar >> buffer;
        {
            std::ofstream ofs(temp_path, std::ios::binary);
            ofs.write(buffer.data(), buffer.size());
        }
        g.loadIndex(temp_path, space);
        boost::filesystem::remove(temp_path);
    }
}

#endif
