#include "LocalOperatorCommon.hpp"

TEST_CASE("Local operator evaluation", "[local_asm]")
{
    // Compare matrix-free evaluation result with multiplication using explicitly constructed operator
    SECTION("Diffusion 2D")
    {
        constexpr auto element = makeQuadElement();
        constexpr auto ET      = element.type;
        constexpr auto EO      = element.order;
        constexpr auto params  = KernelParams{.dimension = 2, .n_equations = 4, .n_unknowns = 3, .n_rhs = 2};
        const auto     phi     = makeSolution< ET, EO, params >(element);
        auto [A, b]            = assembleDiffusionProblem2D< params >(element);
        applyDirichletBCs< ET, EO, params >(A, b, phi);

        auto dom_map = mesh::MeshPartition< EO >::domain_map_t{};
        mesh::pushToDomain(dom_map[0], element);
        const auto global_mesh    = mesh::MeshPartition< EO >{std::move(dom_map), {}};
        using dirichlet_index     = util::smallest_integral_t< operand_size< ET, EO, params > >;
        const auto  local_element = mesh::LocalElementView{element, global_mesh, {}};
        const auto& bnd_node_inds = mesh::ElementTraits< mesh::Element< ET, EO > >::boundary_node_inds;
        auto        dirichlet_bcs = std::vector< dirichlet_index >{};
        for (auto i : bnd_node_inds)
            dirichlet_bcs.push_back(static_cast< dirichlet_index >(i * params.n_unknowns));
        const auto bc_vals = phi(bnd_node_inds, Eigen::all);

        auto x = b;
        x.setRandom();
        auto x_bc = x;
        for (auto i : dirichlet_bcs)
            for (Eigen::Index j = 0; j != params.n_rhs; ++j)
                x_bc(i, j) = 0.;

        auto [diag, rhs] = initDiffusionOperator2D< params >(local_element, dirichlet_bcs, bc_vals);
        auto y           = evalDiffusionOperator2D< params >(local_element, x_bc);

        y(dirichlet_bcs, Eigen::all) = x(dirichlet_bcs, Eigen::all);
        diag(dirichlet_bcs, Eigen::all).setConstant(1.);
        rhs(dirichlet_bcs, Eigen::all) = phi(bnd_node_inds, Eigen::all);

        const auto eval_error = (y - A * x).norm();
        const auto diag_error = (A.diagonal() - diag).norm();
        const auto rhs_error  = (rhs - b).norm();

        constexpr auto eps = 1e-8;
        CHECK(eval_error < eps);
        CHECK(diag_error < eps);
        CHECK(rhs_error < eps);
    }

    SECTION("Diffusion 3D")
    {
        constexpr auto element = makeHexElement();
        constexpr auto ET      = element.type;
        constexpr auto EO      = element.order;
        constexpr auto params  = KernelParams{.dimension = 3, .n_equations = 7, .n_unknowns = 4, .n_rhs = 3};
        const auto     phi     = makeSolution< ET, EO, params >(element);
        auto [A, b]            = assembleDiffusionProblem3D< params >(element);
        applyDirichletBCs< ET, EO, params >(A, b, phi);

        auto dom_map = mesh::MeshPartition< EO >::domain_map_t{};
        mesh::pushToDomain(dom_map[0], element);
        const auto global_mesh    = mesh::MeshPartition< EO >{std::move(dom_map), {}};
        using dirichlet_index     = util::smallest_integral_t< operand_size< ET, EO, params > >;
        const auto  local_element = mesh::LocalElementView{element, global_mesh, {}};
        const auto& bnd_node_inds = mesh::ElementTraits< mesh::Element< ET, EO > >::boundary_node_inds;
        auto        dirichlet_bcs = std::vector< dirichlet_index >{};
        for (auto i : bnd_node_inds)
            dirichlet_bcs.push_back(static_cast< dirichlet_index >(i * params.n_unknowns));
        const auto bc_vals = phi(bnd_node_inds, Eigen::all);

        auto x = b;
        x.setRandom();
        auto x_bc = x;
        for (auto i : dirichlet_bcs)
            for (Eigen::Index j = 0; j != params.n_rhs; ++j)
                x_bc(i, j) = 0.;

        auto [diag, rhs] = initDiffusionOperator3D< params >(local_element, dirichlet_bcs, bc_vals);
        auto y           = evalDiffusionOperator3D< params >(local_element, x_bc);

        y(dirichlet_bcs, Eigen::all) = x(dirichlet_bcs, Eigen::all);
        diag(dirichlet_bcs, Eigen::all).setConstant(1.);
        rhs(dirichlet_bcs, Eigen::all) = phi(bnd_node_inds, Eigen::all);

        const auto eval_error = (y - A * x).norm();
        const auto diag_error = (A.diagonal() - diag).norm();
        const auto rhs_error  = (rhs - b).norm();

        constexpr auto eps = 1e-8;
        CHECK(eval_error < eps);
        CHECK(diag_error < eps);
        CHECK(rhs_error < eps);
    }
}
