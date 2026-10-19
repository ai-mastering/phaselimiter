#include "classic_detector/detector.h"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <gflags/gflags.h>

DEFINE_string(task, "detect2", "Task: detect2 (also accepted as a positional argument).");
DEFINE_string(source, "", "Input audio_analyzer analysis JSON.");
DEFINE_string(detection_preparation, "./resource/detection_preparation.json", "Legacy detector preparation JSON.");
DEFINE_double(neighbor_ratio, 0.03, "Neighbor ratio (Python default 0.03; legacy Rails used 0.01).");
DEFINE_double(outlier_ratio, 0.1, "Outlier fraction for reference eligibility.");

int main(int argc, char **argv) {
    gflags::SetUsageMessage("audio_detector detect2 --source analysis.json --detection_preparation preparation.json");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    try {
        if (FLAGS_task != "detect2" || argc > 2 || (argc == 2 && std::string(argv[1]) != "detect2"))
            throw std::invalid_argument("only the detect2 task is supported");
        if (FLAGS_source.empty()) throw std::invalid_argument("--source is required");
        const auto source = classic_detector::ReadReference(FLAGS_source);
        const auto preparation = classic_detector::ReadPreparation(FLAGS_detection_preparation);
        std::vector<classic_detector::Reference> references;
        references.reserve(preparation.paths.size());
        for (const auto &path : preparation.paths)
            references.push_back(classic_detector::ReadReference(
                classic_detector::ResolveReferencePath(path, FLAGS_detection_preparation)));
        const classic_detector::Detector detector(preparation, references, FLAGS_neighbor_ratio,
                                                  FLAGS_outlier_ratio);
        const auto result = detector.Detect(source);
        std::cout << classic_detector::DetectionJson(result, preparation) << std::endl;
        if (!std::cout) throw std::runtime_error("cannot write detector result");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "audio_detector: " << error.what() << std::endl;
        return 1;
    }
}
