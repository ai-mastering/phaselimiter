## phaselimiter

A high quality audio limiter and an automated mastering algorithm written in C++. 

音圧爆上げくんで使われているリミッターと自動マスタリング

## project status

inactive

- you can create issues and PRs.
- I can respond about once every three months.

## license

MIT

## Installation

todo

prebuilt linux executables are available here

https://github.com/ai-mastering/phaselimiter/releases

## runtime dependencies

- pipenv (python 3)
- ffmpeg (command line program)
- libtbb.so
- libtbbmalloc.so
- libtbbmalloc_proxy.so
- liblapack.so
- libblas.so
- libarmadillo.so
- libsndfile.so

## For developers

### how to build

see CMakeLists
see .circleci/config.yml

### notes

#### windows build

- store prebuilt lib and dll at prebuilt/win64, prebuilt/win32

dynamic link (reason)

- libtbb: shared library is recommended officially
- libtbbmalloc
- libtbbmalloc_proxy
- libsndfile: LGPL
- libpng: probably included in VS redistributable
- libz: probably included in VS redistributable

static link

- libboost_system
- libboost_filesystem
- libboost_serialization
- libboost_math_tr1
- libboost_iostreams
- ippimt.lib
- ippsmt.lib
- ippcoremt.lib
- ippvmmt.lib

libgflags.a




pthread
