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
    MpiComm          comm{MPI_COMM_WORLD};

    constexpr auto problem_def       = std::array{Pair{d_id_t{1}, std::array{false, true, false}},
                                            Pair{d_id_t{2}, std::array{false, true, true}},
                                            Pair{d_id_t{3}, std::array{true, false, false}},
                                            Pair{d_id_t{4}, std::array{true, false, true}},
                                            Pair{d_id_t{5}, std::array{true, true, false}},
                                            Pair{d_id_t{6}, std::array{true, true, true}}};
    constexpr auto problemdef_ctwrpr = ConstexprValue< problem_def >{};

    const auto mesh_full   = comm.getRank() != 0 ? Mesh{} : std::invoke([] {
        constexpr auto       order = 2;
        constexpr std::array dist{0., 1., 2.};
        auto                 retval = makeCubeMesh(dist);
        retval.getPartitions().front().initDualGraph();
        retval.getPartitions().front() = convertMeshToOrder< order >(retval.getPartitions().front());
        return retval;
    });
    const auto mesh_parted = std::invoke([&] { return distributeMesh(comm, mesh_full, {}, problemdef_ctwrpr); });

    const auto cond_map =
        detail::NodeCondensationMap::makeBoundaryNodeCondensationMap(comm, mesh_parted, problemdef_ctwrpr);
    const auto            global_intervals = computeDofIntervals(comm, mesh_parted, cond_map, problemdef_ctwrpr);
    const auto            n_int_global     = global_intervals.size();
    std::vector< size_t > computed_glob_int_sizes(comm.getRank() == 0 ? comm.getSize() : 0);
    comm.gather(std::views::single(n_int_global), computed_glob_int_sizes.begin(), 0);
    if (comm.getRank() == 0)
        REQUIRE(std::ranges::all_of(computed_glob_int_sizes, [&](auto sz) { return sz == n_int_global; }));

    std::vector< unsigned long long > serial_int;
    serial_int.reserve(n_int_global * 3);
    detail::serializeDofIntervals(global_intervals, std::back_inserter(serial_int));
    std::vector< unsigned long long > gather_buf(comm.getRank() == 0 ? serial_int.size() * comm.getSize() : 0);
    std::ranges::generate(gather_buf, [prng = std::mt19937{std::random_device{}()}]() mutable {
        return std::uniform_int_distribution< unsigned long long >{}(prng);
    });
    comm.gather(serial_int, gather_buf.data(), 0);
    if (comm.getRank() == 0)
    {
        for (auto it = gather_buf.begin(); it != gather_buf.end(); std::advance(it, serial_int.size()))
            REQUIRE(std::ranges::equal(std::views::counted(it, serial_int.size()), serial_int));

        const auto  comm_self = MpiComm{MPI_COMM_SELF};
        const auto& full_part = mesh_full.getPartitions().front();
        const auto  cond_map_self =
            detail::NodeCondensationMap::makeBoundaryNodeCondensationMap(comm_self, full_part, problemdef_ctwrpr);
        const auto intervals_expected = detail::computeLocalDofIntervals(full_part, cond_map_self, problemdef_ctwrpr);
        REQUIRE(intervals_expected == global_intervals);
    }
}
