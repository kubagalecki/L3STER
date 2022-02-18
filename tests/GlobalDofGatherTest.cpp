#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/dof/DofIntervals.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/util/GlobalResource.hpp"

#include "TestDataPath.h"

#include <algorithm>
#include <iostream>
#include <random>

int main(int argc, char* argv[])
{
    using namespace lstr;
    GlobalResource< lstr::MpiScopeGuard >::initialize(argc, argv);
    MpiComm comm{};
    try
    {
        std::vector< d_id_t > boundaries(24);
        std::iota(boundaries.begin(), boundaries.end(), 9);
        constexpr auto order = 1;
        const auto     mesh =
            distributeMesh< order >(comm, L3STER_TESTDATA_ABSPATH(gmsh_ascii4_cube_multidom.msh), gmsh_tag, boundaries);
        const auto n_ranks = comm.getSize();
        const auto rank    = comm.getRank();

        constexpr auto problem_def     = ConstexprValue< [] {
            return std::array{Pair{d_id_t{1}, std::array{false, false, false}},
                              Pair{d_id_t{2}, std::array{false, false, true}},
                              Pair{d_id_t{3}, std::array{false, true, false}},
                              Pair{d_id_t{4}, std::array{false, true, true}},
                              Pair{d_id_t{5}, std::array{true, false, false}},
                              Pair{d_id_t{6}, std::array{true, false, true}},
                              Pair{d_id_t{7}, std::array{true, true, false}},
                              Pair{d_id_t{8}, std::array{true, true, true}}};
        }() >{};
        const auto     local_intervals = detail::computeLocalDofIntervals(mesh, problem_def);
        auto           global_data     = detail::gatherGlobalDofIntervals(local_intervals, problem_def, comm);
        auto& [_, global_intervals]    = global_data;
        detail::consolidateDofIntervals(global_intervals);
        const auto            n_int_global = global_intervals.size();
        std::vector< size_t > computed_glob_int_sizes(rank == 0 ? n_ranks : 0);
        comm.gather(&n_int_global, computed_glob_int_sizes.data(), 1, 0);
        if (rank == 0)
            if (std::ranges::any_of(computed_glob_int_sizes, [&n_int_global](auto s) { return s != n_int_global; }))
                throw std::logic_error{"Different ranks have computed different numbers of DOF intervals"};

        std::vector< unsigned long long > serial_int;
        serial_int.reserve(n_int_global * 3);
        detail::serializeDofIntervals(global_intervals, std::back_inserter(serial_int));
        std::vector< unsigned long long > gather_buf(rank == 0 ? serial_int.size() * n_ranks : 0);
        std::ranges::generate(gather_buf, [prng = std::mt19937{std::random_device{}()}]() mutable {
            return std::uniform_int_distribution< unsigned long long >{}(prng);
        });
        comm.gather(serial_int.data(), gather_buf.data(), serial_int.size(), 0);
        if (rank == 0)
            for (auto it = gather_buf.begin(); it != gather_buf.end(); std::advance(it, serial_int.size()))
                if (not std::ranges::equal(std::views::counted(it, serial_int.size()), serial_int))
                    throw std::logic_error{"Different ranks have computed different global DOF intervals"};

        if (rank == 0)
        {
            auto  original_mesh = readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_cube_multidom.msh), gmsh_tag);
            auto& full_part     = original_mesh.getPartitions()[0];
            full_part.initDualGraph();
            full_part                     = convertMeshToOrder< order >(full_part);
            const auto intervals_expected = detail::computeLocalDofIntervals(full_part, problem_def);
            if (intervals_expected != global_intervals)
                throw std::logic_error{
                    "Intervals computed from the distributed mesh differ from those computed from the full mesh"};
        }

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what();
        comm.abort();
        return EXIT_FAILURE;
    }
}
