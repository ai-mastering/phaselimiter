#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "gflags/gflags.h"
#include "sndfile.h"

#include "bakuage/sndfile_wrapper.h"

DEFINE_string(input, "", "Input wave file path.");

int main(int argc, char* argv[]) {
    gflags::SetVersionString("1.0.0");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
   
    // wave読み込み
    bakuage::SndfileWrapper infile;
    SF_INFO sfinfo;

    if ((infile.set(sf_open (FLAGS_input.c_str(), SFM_READ, &sfinfo))) == NULL) {	
        fprintf(stderr, "Not able to open input file %s.\n", FLAGS_input.c_str());
		fprintf(stderr, "%s\n", sf_strerror(NULL));
		return 1;
	} 
    
    printf("{\n");
    printf("  \"channels\": \"%d\",\n", (int)sfinfo.channels);
    printf("  \"format\": \"0x%08x\",\n", (int)sfinfo.format);
    printf("  \"frames\": \"%d\",\n", (int)sfinfo.frames);
    printf("  \"samplerate\": \"%d\",\n", (int)sfinfo.samplerate);
    printf("  \"sections\": \"%d\",\n", (int)sfinfo.sections);
    printf("  \"seekable\": \"%d\"\n", (int)sfinfo.seekable);
    printf("}\n");

	return 0;
}
