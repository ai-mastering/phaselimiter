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

install

- install vc_redist x64
- install ffmpeg
- extract files

dynamic link (reason)

- tbb.dll: shared library is recommended officially
- tbbmalloc.dll
- sndfile.dll: LGPL, stored in prebuilt
- boost: dynamic link is used in conda boost
- zlib.dll: used by boost_iostreams
- zstd.dll: used by boost_iostreams
- libbz2.dll: used by boost_iostreams

static link

- ippimt.lib
- ippsmt.lib
- ippcoremt.lib
- ippvmmt.lib
