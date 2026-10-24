// Compile against the legacy commit documented in hnsw_legacy.txt.
#include <algorithm>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include "hnswlib/hnswlib.h"

class IdSpace: public hnswlib::SpaceInterface<double> {
public:
    std::size_t get_data_size() override { return sizeof(std::int32_t); }
    hnswlib::DISTFUNC<double> get_dist_func() override { return Distance; }
    void *get_dist_func_param() override { return nullptr; }
private:
    static double Distance(const void *a, const void *b, const void *) {
        const auto x = *static_cast<const std::int32_t *>(a);
        const auto y = *static_cast<const std::int32_t *>(b);
        const double dx = (.013 * x + std::sin(.73 * x)) - (.013 * y + std::sin(.73 * y));
        const double dy = (std::cos(.37 * x) + .007 * x) - (std::cos(.37 * y) + .007 * y);
        return std::sqrt(dx * dx + dy * dy);
    }
};

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    IdSpace space;
    hnswlib::HierarchicalNSW<double> index(&space, 96, 8, 200, 100);
    for (std::int32_t id = 0; id < 80; ++id) {
        index.addPoint(&id, (std::uint64_t(1) << 40) + 7 * id);
    }
    index.saveIndex(argv[1]);
}
