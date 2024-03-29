*NOTE* I don't have a Mac to easily test and amend these instructions, but it has been tested through Travis Continuous Integration tool and a lot of people managed to get it working. If anyone wants to edit the wiki with better and more up to date Mac instructions please do!

# Option A

Follow the script [here](https://gitlab.com/Thom/fea_tool/-/blob/master/installer_scripts/macOS/openFace.sh)

All credit goes to @ThomasJanssoone - https://github.com/TadasBaltrusaitis/OpenFace/issues/980

# Option B

## Prerequisites

- I recommend installing [Homebrew](http://brew.sh) as the easiest way to get a variety of Open Source libraries.  Think of it as the Mac equivalent of `apt-get`.  Homebrew usually installs things under `/usr/local/Cellar` and then creates links to the version you're using in `/usr/local/bin`, `/usr/local/lib` etc.  This will be relevant later.

- Install C++17 compiler, *boost*, TBB, dlib, OpenBLAS, and OpenCV, wget (for model download) with:

        brew update
        brew install gcc --HEAD
        brew install boost
        brew install tbb
        brew install openblas
        brew install --build-from-source dlib
        brew install wget
        brew install opencv

- The landmark detection model is not included due to file size, you can download it using the bash `download_models.sh` script. For more details see - https://github.com/TadasBaltrusaitis/OpenFace/wiki/Model-acquisition

- You'll want the [Command Line Tools for Xcode](https://developer.apple.com/downloads/). If you can't use these for any reason, you can build `gcc` using Homebrew -- in fact, it will happen automatically -- but it'll be much slower.

- Get [XQuartz](https://www.xquartz.org) (an X Window system for OS X).  You don't actually need it to *run* OpenFace, but having the X libraries and include files on your system will make OpenFace (and various other things) much easier to build.

- Optional, but you may find it useful to tell CMake how to find your X Windows libraries, since they may not be in the same place as expected on Linux.  Add the following lines to CMakeLists.txt (e.g. after the similar section for OpenCV):

        find_package( X11 REQUIRED )
        MESSAGE("X11 information:")
        MESSAGE("  X11_INCLUDE_DIR: ${X11_INCLUDE_DIR}")
        MESSAGE("  X11_LIBRARIES: ${X11_LIBRARIES}")
        MESSAGE("  X11_LIBRARY_DIRS: ${X11_LIBRARY_DIRS}")
        include_directories( ${X11_INCLUDE_DIR} )

## Building

After that, the build process is very similar to Linux (in OpenFace directory execute the following).

    mkdir build
    cd build
    cmake -D CMAKE_BUILD_TYPE=RELEASE ..  
    make

and you should have binaries in the `bin` directory. e.g. you can run:

    build/bin/FaceLandmarkVid -device 0

## OpenBLAS performance

OpenFace uses OpenBLAS to accelerate numerical computations and TBB for parallelization, in some cases the threading of OpenBLAS and TBB clash. This can lead to the following error:
`OpenBLAS : Program will terminate because you tried to start too many threads.`

To fix it and to potentially improve OpenFace performance:

You can add the below environmental variables (run in the shell just before running any of the OpenFace executables):

    export OMP_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1

You can reinstall OpenBLAS with openmp:

    brew reinstall openblas —with openmp

Also see this thread: https://github.com/TadasBaltrusaitis/OpenFace/issues/748