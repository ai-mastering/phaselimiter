#ifndef BAKUAGE_BAKUAGE_FFMPEG_H_
#define BAKUAGE_BAKUAGE_FFMPEG_H_

#include <iostream>
#include <stdexcept>
#include <sstream>
#include <mutex>
//#include "process.hpp"

namespace bakuage {

class FFMpeg {
public:
    static void Execute(const std::string &ffmpeg_path, const std::string &input_filename, const std::string &output_filename,
                 const std::string &options) {
        std::stringstream command, out, err;
        std::mutex m;

        command << ffmpeg_path << " -i \"" << input_filename << "\" "
            << options << " \"" << output_filename << "\"";

        int exit_status = std::system(command.str().c_str());



/*
        // Processの後にmが解放されるようにする
        int exit_status = 0;
        {
            std::cerr << "process " << command.str() << std::endl;
            Process process(command.str(), ""
            );
            std::cerr << "process called" << std::endl;

            if (process.get_id() == 0) {
                throw std::logic_error("ffmpeg start failed");
            }
            std::cerr << "id" << process.get_id() << std::endl;
            process.close_stdin();
            std::cerr << "stdin closed" << std::endl;

            exit_status = process.get_exit_status();
            std::cerr << "exit_status" << exit_status << std::endl;
        }
        */
        if (exit_status) {
            std::lock_guard<std::mutex> lock(m);
            std::stringstream message;
            message << "ffmpeg unsuccessful terminated exit status: " << exit_status
            ;
//                << ", stdout: " << out.str() << ", stderr: " << err.str();
            throw std::logic_error(message.str());
        }
    }
};

}


#endif
