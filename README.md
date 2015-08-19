# Yggdrasil Demo

## Quick Guide


## CUDA


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
