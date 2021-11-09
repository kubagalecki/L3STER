#ifndef L3STER_BENCHMARKS_COMMON_HPP
#define L3STER_BENCHMARKS_COMMON_HPP

#include "benchmark/benchmark.h"

#include "l3ster/l3ster.hpp"

using namespace lstr;

inline auto getExampleHexElement()
{
    Element< ElementTypes::Hex, 1 > element{{0, 1, 2, 3, 4, 5, 6, 7},
                                            ElementData< ElementTypes::Hex, 1 >{{Point{0., 0., 0.},
                                                                                 Point{1., 0., 0.},
                                                                                 Point{0., 1., 0.},
                                                                                 Point{1., 1., 0.},
                                                                                 Point{0., 0., 1.},
                                                                                 Point{1., 0., 1.},
                                                                                 Point{0., 1., 1.},
                                                                                 Point{2., 2., 2.}}},
                                            0};
    return element;
}
#endif // L3STER_BENCHMARKS_COMMON_HPP
