#pragma once

namespace bakuage {
namespace loudness_contours {

double SplToPhon(double hz, double spl);
double HzToSplAt60Phon(double hz);

// loudness曲線とは上下逆なので注意
    double HzToYoutubeWeighting(double hz);
    
}
}


