#ifndef BAKUAGE_BAKUAGE_FILE_UTILS_H_
#define BAKUAGE_BAKUAGE_FILE_UTILS_H_

#include <string>
#include <vector>

namespace bakuage {

// Publish through a sibling temporary file, preserving the destination on failure.
void PublishFile(const std::string &source, const std::string &destination);

class TemporaryFiles {
public:
    TemporaryFiles(const std::string &_directory);
    virtual ~TemporaryFiles();

    std::string UniquePath(const std::string &extension);
private:
    static int Seed();
    static char RandomChar();

    std::vector<std::string> temporaries_;
    std::string directory_;
};
}

#endif
