//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// Decrement iterator prior to begin.

// UNSUPPORTED: libcxx-no-debug-mode

#define _LIBCPP_DEBUG 1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <vector>
#include <cassert>
#include <iterator>
#include <exception>
#include <cstdlib>

#include "test_macros.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
    typedef int T;
    typedef std::vector<T> C;
    C c(1);
    C::iterator i = c.end();
    --i;
    assert(i == c.begin());
    --i;
    assert(false);
    }
#if TEST_STD_VER >= 11
    {
    typedef int T;
    typedef std::vector<T, min_allocator<T>> C;
    C c(1);
    C::iterator i = c.end();
    --i;
    assert(i == c.begin());
    --i;
    assert(false);
    }
#endif
}
