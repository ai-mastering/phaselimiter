#!/bin/bash

set -ex

cd $(mktemp -d)
git clone -b master --depth 1 --single-branch https://github.com/ai-mastering/phaselimiter.git phaselimiter

sudo mkdir -p /etc/phaselimiter
sudo cp -R phaselimiter/* /etc/phaselimiter/
sudo cp phaselimiter/.python-version /etc/phaselimiter/

cd /etc/phaselimiter/
sudo chmod +x built/*
sudo cp built/* /usr/local/bin
sudo chmod +x script/audio_detector
sudo cp script/audio_detector /usr/local/bin/

pyenv exec pipenv install
