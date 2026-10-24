#include "bakuage/utils.h"

#include <locale>
#ifdef _WIN32
#include <codecvt>
#endif
#include <boost/iostreams/device/mapped_file.hpp>

namespace bakuage {

#ifdef _WIN32
// Windows paths use UTF-16, including surrogate pairs for non-BMP characters.
std::string WStringToString(const std::wstring &w) {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>,wchar_t> cv;
    return cv.to_bytes(w);
}

std::wstring StringToWString(const std::string &s) {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>,wchar_t> cv;
    return cv.from_bytes(s);
}

// http://sayahamitt.net/utf8%E3%81%AAstring%E5%85%A5%E3%82%8C%E3%81%9F%E3%82%89shiftjis%E3%81%AAstring%E5%87%BA%E3%81%A6%E3%81%8F%E3%82%8B%E9%96%A2%E6%95%B0%E4%BD%9C%E3%81%A3%E3%81%9F/
//std::string UTF8toSjis(std::string srcUTF8) {
//}

template <>
std::string NormalizeToString<std::wstring>(const std::wstring &input) {
	return WStringToString(input);
}
#endif

template <>
std::string NormalizeToString<std::string>(const std::string &input) {
	return input;
}

    std::string LoadStrFromFile(const char *path) {
        boost::iostreams::mapped_file mmap(path, boost::iostreams::mapped_file::readonly);
        const std::string str(mmap.const_data(), mmap.const_data() + mmap.size());
        return str;
    }

    void LoadDataFromFile(const char *path, const std::function<void (const char *, size_t)> &write) {
        boost::iostreams::mapped_file mmap(path, boost::iostreams::mapped_file::readonly);
        write(mmap.const_data(), mmap.size());
    }

}
