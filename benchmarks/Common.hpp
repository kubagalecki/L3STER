#ifndef L3STER_BENCHMARKS_COMMON_HPP
#define L3STER_BENCHMARKS_COMMON_HPP

#include "benchmark/benchmark.h"

#include "l3ster/l3ster.hpp"

using namespace lstr;

template < el_o_t O >
auto getExampleHexElement()
{
    using el_t = Element< ElementTypes::Hex, O >;
    std::array< n_id_t, el_t::n_nodes > nodes;
    std::iota(begin(nodes), end(nodes), 0);
    el_t element{nodes,
                 ElementData< ElementTypes::Hex, O >{{Point{0., 0., 0.},
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
