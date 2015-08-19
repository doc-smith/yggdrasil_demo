#include "sum.h"

#include <yggdrasil/contrib/gtest/include/gtest/gtest.h>


TEST(TestSum, MyTestSum) {
    EXPECT_EQ(sum(3, 2), 5);
}

