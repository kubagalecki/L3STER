#include "LocalOperatorCommon.hpp"

template < typename MakeElement >
static void diff2DTest(MakeElement&& make_element)
{
    const auto element    = std::invoke(std::forward< MakeElement >(make_element));
    using eltype          = std::decay_t< decltype(element) >;
    constexpr auto ET     = eltype::type;
    constexpr auto EO     = eltype::order;
    constexpr auto params = KernelParams{.dimension = 2, .n_equations = 4, .n_unknowns = 3, .n_rhs = 2};
    const auto     phi    = makeSolution< ET, EO, params >(element);
    auto [A, b]           = assembleDiffusionProblem2D< params >(element);
    applyDirichletBCs< ET, EO, params >(A, b, phi);

    auto dom_map = typename MeshPartition< EO >::domain_map_t{};
    pushToDomain(dom_map[0], element);
    const auto global_mesh    = MeshPartition< EO >{std::move(dom_map), {}};
    using dirichlet_index     = util::smallest_integral_t< operand_size< ET, EO, params > >;
    const auto  local_element = LocalElementView{element, global_mesh, {}};
    const auto& bnd_node_inds = ElementTraits< Element< ET, EO > >::boundary_node_inds;
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

template < typename MakeElement >
static void diff3DTest(MakeElement&& make_element)
{
    const auto element    = std::invoke(std::forward< MakeElement >(make_element));
    using eltype          = std::decay_t< decltype(element) >;
    constexpr auto ET     = eltype::type;
    constexpr auto EO     = eltype::order;
    constexpr auto params = KernelParams{.dimension = 3, .n_equations = 7, .n_unknowns = 4, .n_rhs = 3};
    const auto     phi    = makeSolution< ET, EO, params >(element);
    auto [A, b]           = assembleDiffusionProblem3D< params >(element);
    applyDirichletBCs< ET, EO, params >(A, b, phi);

    auto dom_map = typename MeshPartition< EO >::domain_map_t{};
    pushToDomain(dom_map[0], element);
    const auto global_mesh    = MeshPartition< EO >{std::move(dom_map), {}};
    using dirichlet_index     = util::smallest_integral_t< operand_size< ET, EO, params > >;
    const auto  local_element = LocalElementView{element, global_mesh, {}};
    const auto& bnd_node_inds = ElementTraits< Element< ET, EO > >::boundary_node_inds;
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

// Compare matrix-free evaluation result with multiplication using explicitly constructed operator
TEST_CASE("Local operator evaluation", "[local_asm]")
{
    SECTION("Diffusion 2D, GO=1")
    {
        diff2DTest([] { return makeQuadElement(); });
    }

    SECTION("Diffusion 3D, GO=1")
    {
        diff3DTest([] { return makeHexElement(); });
    }

    SECTION("Diffusion 2D, GO=2")
    {
        diff2DTest([] { return makeQuad2Element(); });
    }

    SECTION("Diffusion 3D, GO=2")
    {
        diff3DTest([] { return makeHex2Element(); });
    }
}
