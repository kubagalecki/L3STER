#include "l3ster/post/VtkExport.hpp"
#include "l3ster/assembly/ComputeValuesAtNodes.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/util/GlobalResource.hpp"

#include <numbers>

using namespace lstr;
using namespace std::numbers;

void run2D()
{
    const MpiComm comm;

    const auto node_distx = [] {
        std::array< double, 13 > retval;
        for (size_t i = 0; i < retval.size(); ++i)
            retval[i] = -.5 + i * 2.5 / (retval.size() - 1);
        return retval;
    }();
    const auto node_disty = [] {
        std::array< double, 21 > retval;
        for (size_t i = 0; i < retval.size(); ++i)
            retval[i] = -.5 + i * 2. / (retval.size() - 1);
        return retval;
    }();

    Mesh mesh;
    if (comm.getRank() == 0)
    {
        constexpr auto mesh_order = 2;
        mesh                      = makeSquareMesh(node_distx, node_disty);
        mesh.getPartitions()[0].initDualGraph();
        mesh.getPartitions()[0] = convertMeshToOrder< mesh_order >(mesh.getPartitions()[0]);
    }
    constexpr d_id_t domain_id      = 0;
    constexpr d_id_t bot_boundary   = 1;
    constexpr d_id_t top_boundary   = 2;
    constexpr d_id_t left_boundary  = 3;
    constexpr d_id_t right_boundary = 4;
    const auto my_partition = distributeMesh(comm, mesh, {bot_boundary, top_boundary, left_boundary, right_boundary});

    constexpr auto problem_def =
        ConstexprValue< std::array{Pair{d_id_t{domain_id}, std::array{false, true, false, true, true}},
                                   Pair{d_id_t{bot_boundary}, std::array{true, false, true, false, false}},
                                   Pair{d_id_t{top_boundary}, std::array{true, false, true, false, false}}} >{};
    const auto dof_intervals  = computeDofIntervals(my_partition, problem_def, comm);
    const auto map            = NodeToDofMap{my_partition, dof_intervals};
    const auto sparsity_graph = detail::makeSparsityGraph(my_partition, problem_def, dof_intervals, comm);

    const auto node_vals =
        makeTeuchosRCP< Tpetra::FEMultiVector<> >(sparsity_graph->getRowMap(), sparsity_graph->getImporter(), 1u);

    auto exporter = PvtuExporter{my_partition,
                                 map,
                                 *node_vals->getMap(),
                                 std::vector< std::string >{"C1", "Cpi", "vel"},
                                 std::array< unsigned char, 5 >{0, 2, 1, 2, 2}};

    node_vals->switchActiveMultiVector();
    node_vals->beginModify();
    computeValuesAtNodes< std::array< size_t, 2 >{0, 2} >(
        [&](const SpaceTimePoint& p) {
            Eigen::Vector2d retval;
            retval[0] = 1.;
            retval[1] = pi;
            return retval;
        },
        my_partition,
        std::array{bot_boundary, top_boundary},
        map,
        *node_vals->getVectorNonConst(0));
    const double Re     = 40.;
    const double lambda = Re / 2. - std::sqrt(Re * Re / 4. - 4. * pi * pi);
    computeValuesAtNodes< std::array< size_t, 3 >{1, 3, 4} >(
        [&](const SpaceTimePoint& p) {
            Eigen::Vector3d retval;
            // Kovasznay flow velocity field
            retval[0] = 1. - std::exp(lambda * p.space.x()) * std::cos(2. * pi * p.space.y());
            retval[1] = lambda * std::exp(lambda * p.space.x()) * std::sin(2 * pi * p.space.y()) / (2. * pi);
            retval[2] = 0.;
            return retval;
        },
        my_partition,
        std::views::single(domain_id),
        map,
        *node_vals->getVectorNonConst(0));
    node_vals->endModify();
    node_vals->doOwnedToOwnedPlusShared(Tpetra::REPLACE);
    node_vals->switchActiveMultiVector();
    node_vals->sync_host();
    exporter.exportResults("test_results_2D", comm, *node_vals->getVector(0));
}

void run3D()
{
    const MpiComm comm;

    const auto node_dist = [] {
        std::array< double, 11 > retval;
        for (size_t i = 0; i < retval.size(); ++i)
            retval[i] = -1. + i * 2. / (retval.size() - 1);
        return retval;
    }();

    Mesh mesh;
    if (comm.getRank() == 0)
    {
        constexpr auto mesh_order = 2;
        mesh                      = makeCubeMesh(node_dist);
        mesh.getPartitions()[0].initDualGraph();
        mesh.getPartitions()[0] = convertMeshToOrder< mesh_order >(mesh.getPartitions()[0]);
    }
    const auto my_partition = distributeMesh(comm, mesh, {1, 2, 3, 4, 5, 6});

    constexpr auto problem_def    = ConstexprValue< std::array{Pair{d_id_t{0}, std::array{true, true, true}}} >{};
    const auto     dof_intervals  = computeDofIntervals(my_partition, problem_def, comm);
    const auto     map            = NodeToDofMap{my_partition, dof_intervals};
    const auto     sparsity_graph = detail::makeSparsityGraph(my_partition, problem_def, dof_intervals, comm);

    const auto node_vals =
        makeTeuchosRCP< Tpetra::FEMultiVector<> >(sparsity_graph->getRowMap(), sparsity_graph->getImporter(), 1u);

    auto exporter = PvtuExporter{my_partition,
                                 map,
                                 *node_vals->getMap(),
                                 std::vector< std::string >{"vec3D"},
                                 std::array< unsigned char, 3 >{0, 0, 0}};

    node_vals->switchActiveMultiVector();
    node_vals->beginModify();
    computeValuesAtNodes< std::array< size_t, 3 >{0, 1, 2} >(
        [&](const SpaceTimePoint& point) {
            Eigen::Vector3d retval;
            const auto&     p = point.space;
            const auto      r = std::sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z());
            retval[0]         = r;
            retval[1]         = p.y();
            retval[2]         = p.z();
            return retval;
        },
        my_partition,
        std::views::single(0),
        map,
        *node_vals->getVectorNonConst(0));
    node_vals->endModify();
    node_vals->doOwnedToOwnedPlusShared(Tpetra::REPLACE);
    node_vals->switchActiveMultiVector();
    node_vals->sync_host();
    exporter.exportResults("test_results_3D", comm, *node_vals->getVector(0));
}

int main(int argc, char* argv[])
{
    GlobalResource< MpiScopeGuard >::initialize(argc, argv);

    run2D();
    run3D();

    // TODO: Programatically check whether the data was exported correctly. For the time being, check manually...
}