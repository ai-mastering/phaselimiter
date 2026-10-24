#ifndef bakuage_hnsw_serialization_h
#define bakuage_hnsw_serialization_h

#include <cstddef>
#include <fstream>
#include <memory>
#include <boost/filesystem.hpp>
#include "hnswlib/hnswlib.h"
#include "bakuage/memory.h"
#include "bakuage/utils.h"

namespace bakuage {
    // Legacy indices used explicit 64-bit sizes and labels. Keep that layout
    // when reading them with upstream hnswlib on x86-64 and ARM64.
    static_assert(sizeof(std::size_t) == 8 && sizeof(hnswlib::labeltype) == 8,
                  "Saved HNSW indices require a 64-bit build");
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
    void LoadHnswFromArchive(Archive & ar, std::unique_ptr<hnswlib::HierarchicalNSW<T>> &g, const unsigned int version, hnswlib::SpaceInterface<T> *space) {
        const auto temp_path = NormalizeToString((boost::filesystem::temp_directory_path() / boost::filesystem::unique_path()).native());
        bakuage::AlignedPodVector<char> buffer;
        ar >> buffer;
        {
            std::ofstream ofs(temp_path, std::ios::binary);
            ofs.write(buffer.data(), buffer.size());
        }
        // Replace the existing graph only after a fresh one loads successfully.
        g.reset(new hnswlib::HierarchicalNSW<T>(space, temp_path));
        boost::filesystem::remove(temp_path);
    }
}

#endif
