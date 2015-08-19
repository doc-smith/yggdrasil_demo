# Yggdrasil Demo

## Quick Guide

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
