#ifndef BAKUAGE_BAKUAGE_FILE_UTILS_H_
#define BAKUAGE_BAKUAGE_FILE_UTILS_H_

#include <string>
#include <vector>

namespace bakuage {

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
