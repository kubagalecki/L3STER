#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/dofs/DofIntervals.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Common.hpp"

#include <algorithm>
#include <random>

using namespace lstr;
using namespace lstr::dofs;

template < CondensationPolicy CP >
void test(CondensationPolicyTag< CP > = {})
{
    const auto comm = MpiComm{MPI_COMM_WORLD};

    constexpr auto problem_def       = ProblemDef{L3STER_DEFINE_DOMAIN(1, 1),
                                            L3STER_DEFINE_DOMAIN(2, 1, 2),
                                            L3STER_DEFINE_DOMAIN(3, 0),
                                            L3STER_DEFINE_DOMAIN(4, 0, 2),
                                            L3STER_DEFINE_DOMAIN(5, 0, 1),
                                            L3STER_DEFINE_DOMAIN(6, 0, 1, 2)};
    constexpr auto problemdef_ctwrpr = util::ConstexprValue< problem_def >{};

    static constexpr el_o_t mesh_order     = 2;
    const auto              mesh_full      = comm.getRank() == 0 ? std::invoke([] {
        constexpr std::array dist{0., 1., 2.};
        auto                 retval = mesh::makeCubeMesh(dist);
        return mesh::convertMeshToOrder< mesh_order >(retval);
    })
                                                                 : mesh::MeshPartition< mesh_order >{};
    auto                    mesh_full_copy = copy(mesh_full);
    const auto              mesh_parted    = distributeMesh(comm, std::move(mesh_full_copy), problemdef_ctwrpr);

    const auto cond_map                = makeCondensationMap< CP >(comm, *mesh_parted, problemdef_ctwrpr);
    const auto global_intervals        = computeDofIntervals(comm, *mesh_parted, cond_map, problemdef_ctwrpr);
    const auto n_int_global            = global_intervals.size();
    auto       computed_glob_int_sizes = std::vector< size_t >(comm.getRank() == 0 ? comm.getSize() : 0);
    comm.gather(std::views::single(n_int_global), computed_glob_int_sizes.begin(), 0);
    if (comm.getRank() == 0)
        REQUIRE(std::ranges::all_of(computed_glob_int_sizes, [&](auto sz) { return sz == n_int_global; }));

    std::vector< unsigned long long > serial_int;
    serial_int.reserve(n_int_global * 3);
    serializeDofIntervals(global_intervals, std::back_inserter(serial_int));
    std::vector< unsigned long long > gather_buf(comm.getRank() == 0 ? serial_int.size() * comm.getSize() : 0);
    std::ranges::generate(gather_buf, [prng = std::mt19937{std::random_device{}()}]() mutable {
        return std::uniform_int_distribution< unsigned long long >{}(prng);
    });
    comm.gather(serial_int, gather_buf.data(), 0);
    if (comm.getRank() == 0)
    {
        for (auto it = gather_buf.begin(); it != gather_buf.end(); std::advance(it, serial_int.size()))
            REQUIRE(std::ranges::equal(std::views::counted(it, serial_int.size()), serial_int));

        const auto comm_self          = MpiComm{MPI_COMM_SELF};
        const auto cond_map_self      = makeCondensationMap< CP >(comm_self, mesh_full, problemdef_ctwrpr);
        const auto intervals_expected = computeLocalDofIntervals(mesh_full, cond_map_self, problemdef_ctwrpr);
        REQUIRE(intervals_expected == global_intervals);
    }
}

int main(int argc, char* argv[])
{
    const auto max_par_guard = util::MaxParallelismGuard{4};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};
    test< CondensationPolicy::None >();
    test< CondensationPolicy::ElementBoundary >();
}
