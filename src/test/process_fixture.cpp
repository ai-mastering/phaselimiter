#include <fstream>
#include <string>

// Record the exact arguments received by a process launched as FFmpeg.
int main(int argc, char **argv) {
    if (argc < 5) return 2;
    if (std::string(argv[3]) == "fail") return 23;
    std::ofstream output(argv[argc - 1], std::ios::binary);
    for (int i = 1; i < argc - 1; ++i) output << argv[i] << '\n';
    return output ? 0 : 3;
}
