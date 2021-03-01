#include <iostream>

void TestPerfectHashPowerOf2() {
    for (uint32_t i = 0; i < 1000 * 1000 * 1000; i++) {
        uint32_t y = 2 * i + 1;
        int flags[32] = { 0 };
        int ok = 1;
        for (uint32_t j = 0; j < 32; j++) {
            uint32_t x = (uint32_t)1 << j;
            uint32_t h = ((y * x) >> (32 - 5)) & 31;
            flags[h]++;
            if (flags[h] > 1) {
                ok = 0;
                break;
            }
        }
        if (ok) {
            std::cout << "found " << y << std::endl;
            return;
        }
    }
    std::cout << "not found" << std::endl;
}
