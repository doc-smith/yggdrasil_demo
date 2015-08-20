# Yggdrasil Demo

[Yggdrasil](https://github.com/drsmithization/yggdrasil) is a C++ oriented cmake-based build infrastructure adapted from https://github.com/yandex/balancer. This project is an usage example of Yggdrasil.

## Quick Guide
For using Yggdrasil in your C++ repository you need to add [the main Yggdrasil repo](https://github.com/drsmithization/yggdrasil) as a submodule (or just put the code into the subdirectory **yggdrasil**) and create a very simple root CMakeLists.txt ([example](https://github.com/drsmithization/yggdrasil_demo/blob/master/CMakeLists.txt)). *ROOT_PROJECTS* specify a list of projects that will be built by default (without specifying BUILD_ONLY).

In most cases you will need only three basic templates: for executable programs, static/dynamic libraries and for specifying subprojects in subdirectories.
Here is a basic template for executable targets:

```cmake
PROGRAM()

# specify all project .cpp files here
SRCS (
    a.cpp
    b.cpp
    c.cu # Yggdrasil supports CUDA
    main.cpp
)

# dependencies
PEERDIR (
    libs/x
    libs/y
    
    # add ADDINCL if you don't want to build a library
    #    (only modify include paths)
    ADDINCL libs/z
)

END()
```

Static library template is very similar to PROGRAM (see [library/sum/CMakeLists.txt](https://github.com/drsmithization/yggdrasil_demo/blob/master/library/sum/CMakeLists.txt)):
```cmake
LIBRARY()

SRCS (
    ...
)

PEERDIR (
    ...
)

END()
```

And if you just want to list subprojects:
(see [library/CMakeLists.txt](https://github.com/drsmithization/yggdrasil_demo/blob/master/library/CMakeLists.txt))
```cmake
RECURSE (
    subprojects
)
```

## local.cmake
You can create the local.cmake file in the build directory to control generation process using predefined options. For example:
```
SET(MY_GCC gcc-5.0)
SET(MY_GPP g++-5.0)
SET(USE_STATIC_CPP_RUNTIME no)
ENABLE(NOSSE)
```

## Quick Examples

Build and run unit tests from library/x:
```bash
$ cmake -DUT_PERDIR ../
$ make library-x_ut
$ bin/library-x_ut
```

Release build (default is Debug):
```bash
$ cmake -DCMAKE_BUILD_TYPE=Release ../
```

Build only the project projects/xproj
```bash
$ cmake -DCMAKE_BUILD_ONLY=projects/xproj ../
```

## CUDA
See [projects/cudatest](https://github.com/drsmithization/yggdrasil_demo/tree/master/projects/cudatest)


## Unit Testing
For each project Yggdrasil automatically discovers all files with the **_ut.cpp** prefix and creates a special executable target from these files and gtest/src/gtest_main.cc (see [ut_template.cmake](https://github.com/drsmithization/yggdrasil/blob/master/cmake/include/ut_template.cmake)). 

Here is an example of code for testing our outstanding summation library:
```C++
#include "sum.h"

#include <yggdrasil/contrib/gtest/include/gtest/gtest.h>


TEST(TestSum, MyTestSum) {
    EXPECT_EQ(sum(3, 2), 5);
}
```

```bash
$ cd build
$ cmake -DUT_PERDIR=yes ../
$ make library-sum_ut
```
