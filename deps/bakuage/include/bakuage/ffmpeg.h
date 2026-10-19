#ifndef BAKUAGE_BAKUAGE_FFMPEG_H_
#define BAKUAGE_BAKUAGE_FFMPEG_H_

#include <boost/process.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace bakuage {
class FFMpeg {
public:
    static void Execute(const std::string &ffmpeg_path, const std::string &input_filename,
                        const std::string &output_filename, const std::vector<std::string> &options) {
        // Resolve bare executable names through PATH without invoking a shell.
        boost::filesystem::path executable(ffmpeg_path);
        if (!executable.has_parent_path()) executable = boost::process::search_path(ffmpeg_path);
        if (executable.empty()) throw std::runtime_error("ffmpeg executable not found: " + ffmpeg_path);
        std::vector<std::string> args{"-nostdin", "-i", input_filename};
        args.insert(args.end(), options.begin(), options.end());
        // An absolute path cannot be interpreted as a command-line option.
        args.push_back(boost::filesystem::absolute(output_filename).string());
        boost::process::child process(executable, boost::process::args(args));
        process.wait();
        if (process.exit_code() != 0) {
            throw std::runtime_error("ffmpeg terminated with exit status: " + std::to_string(process.exit_code()));
        }
    }
};
}
#endif
