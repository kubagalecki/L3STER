#include "l3ster/assembly/DofIntervals.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Common.hpp"

#include <algorithm>
#include <random>

int main(int argc, char* argv[])
{
    using namespace lstr;
    L3sterScopeGuard scope_guard{argc, argv};
    MpiComm          comm{};

    constexpr auto problem_def = ConstexprValue< std::array{Pair{d_id_t{1}, std::array{false, true, false}},
                                                            Pair{d_id_t{2}, std::array{false, true, true}},
                                                            Pair{d_id_t{3}, std::array{true, false, false}},
                                                            Pair{d_id_t{4}, std::array{true, false, true}},
                                                            Pair{d_id_t{5}, std::array{true, true, false}},
                                                            Pair{d_id_t{6}, std::array{true, true, true}}} >{};

    const auto n_ranks = comm.getSize();
    const auto my_rank = comm.getRank();

    const auto            mesh_full = my_rank != 0 ? Mesh{} : std::invoke([] {
        constexpr auto       order = 2;
        constexpr std::array dist{0., 1., 2.};
        auto                 retval = makeCubeMesh(dist);
        retval.getPartitions().front().initDualGraph();
        retval.getPartitions().front() = convertMeshToOrder< order >(retval.getPartitions().front());
        return retval;
    });
    std::vector< d_id_t > boundaries(6);
    std::iota(boundaries.begin(), boundaries.end(), 1);
    const auto mesh_parted = distributeMesh(comm, mesh_full, boundaries, problem_def);

    const auto local_intervals  = detail::computeLocalDofIntervals(mesh_parted, problem_def);
    auto       global_data      = detail::gatherGlobalDofIntervals(local_intervals, problem_def, comm);
    auto& [_, global_intervals] = global_data;
    detail::consolidateDofIntervals(global_intervals);
    const auto            n_int_global = global_intervals.size();
    std::vector< size_t > computed_glob_int_sizes(my_rank == 0 ? n_ranks : 0);
    comm.gather(&n_int_global, computed_glob_int_sizes.data(), 1, 0);
    if (my_rank == 0)
        REQUIRE(std::ranges::none_of(computed_glob_int_sizes, [&](auto s) { return s != n_int_global; }));

    std::vector< unsigned long long > serial_int;
    serial_int.reserve(n_int_global * 3);
    detail::serializeDofIntervals(global_intervals, std::back_inserter(serial_int));
    std::vector< unsigned long long > gather_buf(my_rank == 0 ? serial_int.size() * n_ranks : 0);
    std::ranges::generate(gather_buf, [prng = std::mt19937{std::random_device{}()}]() mutable {
        return std::uniform_int_distribution< unsigned long long >{}(prng);
    });
    comm.gather(serial_int.data(), gather_buf.data(), serial_int.size(), 0);
    if (my_rank == 0)
    {
        for (auto it = gather_buf.begin(); it != gather_buf.end(); std::advance(it, serial_int.size()))
            REQUIRE(std::ranges::equal(std::span{it, serial_int.size()}, serial_int));

        const auto& full_part          = mesh_full.getPartitions().front();
        const auto  intervals_expected = detail::computeLocalDofIntervals(full_part, problem_def);
        REQUIRE(intervals_expected == global_intervals);
    }
}
