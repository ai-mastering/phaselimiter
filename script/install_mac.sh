#!/bin/bash

echo 'run this script in root of phaselimiter repository'
echo 'close xcode before run this script'

rm -rf CMakeFiles CMakeCache.txt deps/gflags/CMakeFiles \
  && cmake -GXcode -DCMAKE_BUILD_TYPE=Release .

xcodebuild \
    -project phaselimiter.xcodeproj \
    -configuration Release

sudo mkdir -p /etc/phaselimiter
sudo cp -R ./* /etc/phaselimiter/
sudo cp ./.python-version /etc/phaselimiter/

(
cd /etc/phaselimiter/
sudo chmod +x bin/Release/*
sudo cp bin/Release/* /usr/local/bin
sudo chmod +x script/audio_detector
sudo cp script/audio_detector /usr/local/bin/
sudo chmod 777 ./
pyenv exec pipenv install
)
