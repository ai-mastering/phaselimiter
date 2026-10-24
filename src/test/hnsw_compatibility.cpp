#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <sstream>
#include <vector>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <gtest/gtest.h>
#include "bakuage/hnsw_serialization.h"

namespace {
using Id = std::int32_t;
using Index = hnswlib::HierarchicalNSW<double>;
using Neighbor = std::pair<double, hnswlib::labeltype>;

hnswlib::labeltype Label(Id id) {
    return (std::uint64_t(1) << 40) + 7 * id;
}

double Distance(Id a, Id b) {
    const double dx = (.013 * a + std::sin(.73 * a)) - (.013 * b + std::sin(.73 * b));
    const double dy = (std::cos(.37 * a) + .007 * a) - (std::cos(.37 * b) + .007 * b);
    return std::sqrt(dx * dx + dy * dy);
}

class IdSpace: public hnswlib::SpaceInterface<double> {
public:
    std::size_t get_data_size() override { return sizeof(Id); }
    hnswlib::DISTFUNC<double> get_dist_func() override { return DistanceFunction; }
    void *get_dist_func_param() override { return nullptr; }
private:
    static double DistanceFunction(const void *a, const void *b, const void *) {
        return Distance(*static_cast<const Id *>(a), *static_cast<const Id *>(b));
    }
};

std::unique_ptr<Index> MakeIndex(IdSpace &space) {
    std::unique_ptr<Index> index(new Index(&space, 96, 8, 200, 100));
    for (Id id = 0; id < 80; ++id) index->addPoint(&id, Label(id));
    return index;
}

void CheckNeighbors(Index &index, std::size_t ef = 10) {
    ASSERT_EQ(80u, index.getCurrentElementCount());
    index.setEf(ef);
    for (Id query: {-5, 0, 17, 79, 81, 94, 121}) {
        std::vector<Neighbor> expected;
        for (Id id = 0; id < 80; ++id) expected.emplace_back(Distance(query, id), Label(id));
        std::sort(expected.begin(), expected.end());
        for (std::size_t k: {1u, 5u, 12u}) {
            SCOPED_TRACE(::testing::Message() << "query=" << query << ", k=" << k);
            auto queue = index.searchKnn(&query, k);
            std::vector<Neighbor> actual;
            while (!queue.empty()) {
                actual.push_back(queue.top());
                queue.pop();
            }
            std::sort(actual.begin(), actual.end());
            ASSERT_EQ(k, actual.size());
            for (std::size_t i = 0; i < k; ++i) {
                EXPECT_EQ(expected[i].second, actual[i].second);
                EXPECT_NEAR(expected[i].first, actual[i].first, 1e-12);
            }
        }
    }
}
}

TEST(HnswCompatibility, LegacyIndexArchiveAndRepeatedReplacement) {
    std::ifstream input(PHASELIMITER_HNSW_LEGACY_INDEX_PATH, std::ios::binary);
    ASSERT_TRUE(input.is_open());
    const std::string bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    ASSERT_FALSE(bytes.empty());
    bakuage::AlignedPodVector<char> buffer(bytes.begin(), bytes.end());
    std::stringstream archive;
    {
        boost::archive::binary_oarchive writer(archive);
        writer << buffer; // Same envelope used inside existing LOF/model caches.
    }
    IdSpace space;
    std::unique_ptr<Index> index(new Index(&space, 1, 8, 200));
    Id stale = -100;
    index->addPoint(&stale, 123);
    for (int repeat = 0; repeat < 3; ++repeat) {
        std::istringstream serialized(archive.str());
        boost::archive::binary_iarchive reader(serialized);
        bakuage::LoadHnswFromArchive(reader, index, 0, &space);
        CheckNeighbors(*index);
        CheckNeighbors(*index, 96);
    }

    // Loading and saving a legacy graph must preserve its binary payload.
    std::stringstream saved;
    {
        boost::archive::binary_oarchive writer(saved);
        bakuage::SaveHnswToArchive(writer, *index, 0);
    }
    bakuage::AlignedPodVector<char> restored;
    boost::archive::binary_iarchive reader(saved);
    reader >> restored;
    ASSERT_EQ(bytes.size(), restored.size());
    EXPECT_TRUE(std::equal(bytes.begin(), bytes.end(), restored.begin()));
}

TEST(HnswCompatibility, NewIndexArchiveRoundTrip) {
    IdSpace space;
    auto original = MakeIndex(space);
    CheckNeighbors(*original);
    std::stringstream archive;
    {
        boost::archive::binary_oarchive writer(archive);
        bakuage::SaveHnswToArchive(writer, *original, 0);
    }
    std::unique_ptr<Index> loaded;
    boost::archive::binary_iarchive reader(archive);
    bakuage::LoadHnswFromArchive(reader, loaded, 0, &space);
    CheckNeighbors(*loaded);
}
