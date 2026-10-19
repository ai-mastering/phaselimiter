#ifndef CLASSIC_DETECTOR_NUMPY_NEIGHBORS_H_
#define CLASSIC_DETECTOR_NUMPY_NEIGHBORS_H_
#include <cstddef>
#include <stdexcept>
#include <vector>
namespace classic_detector {
std::vector<size_t> NumpyNeighbors(const std::vector<double> &distances, size_t count);
}
#endif
