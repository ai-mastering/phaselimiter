#include "classic_detector/detector.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <boost/filesystem.hpp>
#include "picojson.h"

namespace classic_detector {
namespace {
using picojson::value;
using picojson::array;
using picojson::object;
value ReadJson(const std::string &path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open classic JSON: " + path);
    value result;
    const auto error = picojson::parse(result, file);
    if (!error.empty() || !result.is<object>()) throw std::runtime_error("invalid classic JSON: " + path + ": " + error);
    return result;
}
const value &Field(const object &o, const std::string &name) {
    auto it = o.find(name);
    if (it == o.end()) throw std::invalid_argument("classic JSON missing field: " + name);
    return it->second;
}
double Number(const value &v) {
    if (!v.is<double>() || !std::isfinite(v.get<double>())) throw std::invalid_argument("classic JSON requires finite numbers");
    return v.get<double>();
}
const array &Array(const value &v) {
    if (!v.is<array>()) throw std::invalid_argument("classic JSON requires an array");
    return v.get<array>();
}
std::string String(const value &v) {
    if (!v.is<std::string>()) throw std::invalid_argument("classic JSON requires a string");
    return v.get<std::string>();
}
Reference ParseReference(const value &v) {
    const auto &o = v.get<object>();
    const double loudness = Number(Field(o, "loudness"));
    const auto &bands = Array(Field(o, "bands_short"));
    if (bands.empty()) throw std::invalid_argument("classic reference has no bands");
    Reference result;
    const size_t dim = bands.size() * 2;
    result.mean.resize(dim); result.covariance.resize(dim, dim);
    for (size_t i = 0; i < bands.size(); ++i) {
        if (!bands[i].is<object>()) throw std::invalid_argument("invalid classic band");
        result.mean[2 * i] = Number(Field(bands[i].get<object>(), "mid_mean")) - loudness;
        result.mean[2 * i + 1] = Number(Field(bands[i].get<object>(), "side_mean")) - loudness;
    }
    const auto &rows = Array(Field(o, "covariance_short"));
    if (rows.size() != dim) throw std::invalid_argument("invalid classic covariance size");
    for (size_t i = 0; i < dim; ++i) {
        const auto &row = Array(rows[i]);
        if (row.size() != dim) throw std::invalid_argument("invalid classic covariance row size");
        for (size_t j = 0; j < dim; ++j) result.covariance(i, j) = Number(row[j]);
    }
    return result;
}
std::string JsonNumber(double x) {
    // Match Python json.dumps' nonfinite diagnostic values in degenerate LOF
    // populations, rather than silently changing NaN/Infinity into null.
    if (std::isnan(x)) return "NaN";
    if (std::isinf(x)) return x < 0 ? "-Infinity" : "Infinity";
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<double>::max_digits10) << x;
    return stream.str();
}
}
Reference ReadReference(const std::string &path) { return ParseReference(ReadJson(path)); }
Preparation ReadPreparation(const std::string &path) {
    const auto root = ReadJson(path);
    const auto &o = root.get<object>();
    Preparation result;
    result.regularization = Number(Field(o, "covariance_regularization"));
    if (result.regularization < 0) throw std::invalid_argument("negative covariance regularization");
    for (const auto &p : Array(Field(o, "paths"))) result.paths.push_back(String(p));
    const auto &matrix = Array(Field(o, "distance_matrix"));
    if (matrix.size() != result.paths.size()) throw std::invalid_argument("invalid distance matrix size");
    for (const auto &v : matrix) {
        const auto &row = Array(v);
        if (row.size() != result.paths.size()) throw std::invalid_argument("invalid distance matrix row size");
        result.distances.emplace_back();
        for (const auto &x : row) {
            const double distance = Number(x);
            if (distance < 0) throw std::invalid_argument("negative preparation distance");
            result.distances.back().push_back(distance);
        }
    }
    return result;
}
std::string ResolveReferencePath(const std::string &recorded, const std::string &preparation_path) {
    namespace fs = boost::filesystem;
    const fs::path parent = fs::absolute(fs::path(preparation_path)).parent_path();
    std::string portable = recorded;
    std::replace(portable.begin(), portable.end(), '\\', '/');
    const fs::path path(portable);
    if (path.is_absolute()) return recorded;
    // The bundled old files store ./resource/analysis_data/... relative to the
    // project root. The anonymized dataset stores ./analysis_data/... relative
    // to the preparation. Resolve either without depending on the caller's cwd.
    const auto offset = portable.rfind("analysis_data/");
    if (offset != std::string::npos) {
        const auto relocated = parent / portable.substr(offset);
        if (fs::is_regular_file(relocated)) return relocated.string();
    }
    const auto relative = parent / path;
    if (fs::is_regular_file(relative)) return relative.string();
    if (fs::is_regular_file(path)) return recorded;
    throw std::runtime_error("cannot locate detector reference: " + recorded);
}
std::string DetectionJson(const Detection &d, const Preparation &p) {
    return "{\n  \"border_lof\": " + JsonNumber(d.border_lof) +
        ",\n  \"lof\": " + JsonNumber(d.lof) +
        ",\n  \"sound_quality\": " + JsonNumber(d.sound_quality) +
        ",\n  \"ranking\": [{\"path\": " + value(p.paths.at(d.index)).serialize() +
        ", \"lof\": " + JsonNumber(d.reference_lof) + "}]\n}";
}
}
