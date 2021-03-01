#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
docker run -it --rm -v $SCRIPTPATH/../:/phaselimiter contribu/buildenv_docker bash
