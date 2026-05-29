#include "LocalOperatorCommon.hpp"

template < typename MakeElement >
static void diff2DTest(MakeElement&& make_element)
{
    const auto element     = std::invoke(std::forward< MakeElement >(make_element));
    using eltype           = std::decay_t< decltype(element) >;
    constexpr auto ET      = eltype::type;
    constexpr auto EO      = eltype::order;
    constexpr auto n_nodes = ElementTraits< Element< ET, EO > >::nodes_per_element;
    constexpr auto params  = KernelParams{.dimension = 2, .n_equations = 4, .n_unknowns = 3, .n_fields = 1, .n_rhs = 2};

    auto dom_map = typename MeshPartition< EO >::domain_map_t{};
    pushToDomain(dom_map[0], element);
    const auto global_mesh   = MeshPartition< EO >{std::move(dom_map), {}};
    const auto local_element = LocalElementView{element, global_mesh, {}};
    const auto sol_man       = makeRandomlyFilledSolutiondManager(global_mesh, 1);

    const auto x = Operand< params, n_nodes >::Random(operand_size< params, n_nodes >, params.n_rhs).eval();
    const auto y_local_element = evalDiffusionOperatorVar< params >(local_element, x, sol_man, diffusion_kernel_2D_var);
    const auto y_sum_fact      = evalDiffusionVarOperatorSumFact< params >(local_element, x, sol_man);

    constexpr auto eps = 1e-8;
    CHECK((y_local_element - y_sum_fact).norm() < eps);
}

template < typename MakeElement >
static void diff3DTest(MakeElement&& make_element)
{
    const auto element     = std::invoke(std::forward< MakeElement >(make_element));
    using eltype           = std::decay_t< decltype(element) >;
    constexpr auto ET      = eltype::type;
    constexpr auto EO      = eltype::order;
    constexpr auto n_nodes = ElementTraits< Element< ET, EO > >::nodes_per_element;
    constexpr auto params  = KernelParams{.dimension = 3, .n_equations = 7, .n_unknowns = 4, .n_fields = 1, .n_rhs = 2};

    auto dom_map = typename MeshPartition< EO >::domain_map_t{};
    pushToDomain(dom_map[0], element);
    const auto global_mesh   = MeshPartition< EO >{std::move(dom_map), {}};
    const auto local_element = LocalElementView{element, global_mesh, {}};
    const auto sol_man       = makeRandomlyFilledSolutiondManager(global_mesh, 1);

    const auto x = Operand< params, n_nodes >::Random(operand_size< params, n_nodes >, params.n_rhs).eval();
    const auto y_local_element = evalDiffusionOperatorVar< params >(local_element, x, sol_man, diffusion_kernel_3D_var);
    const auto y_sum_fact      = evalDiffusionVarOperatorSumFact< params >(local_element, x, sol_man);

    constexpr auto eps = 1e-8;
    CHECK((y_local_element - y_sum_fact).norm() < eps);
}

// Compare results between the local element approach and the sum-factorization technique
TEST_CASE("Sum-factorized evaluation", "[local_asm]")
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

template < int size, int EO, int QO >
using wrap_test_params = util::ConstexprValue< std::array{size, EO, QO} >;

TEMPLATE_TEST_CASE("Odd-even decomposition",
                   "[local_asm]",
                   (wrap_test_params< 33, 3, 3 >),
                   (wrap_test_params< 33, 3, 4 >),
                   (wrap_test_params< 33, 4, 3 >),
                   (wrap_test_params< 33, 4, 4 >))
{
    constexpr auto params             = TestType::value;
    constexpr auto size               = params[0];
    constexpr auto EO                 = static_cast< el_o_t >(params[1]);
    constexpr auto QO                 = static_cast< q_o_t >(params[2]);
    constexpr auto sum_fact_params_sf = SumFactParams{.basis_order  = EO,
                                                      .quad_order   = QO,
                                                      .basis_type   = basis::BasisType::Lagrange,
                                                      .quad_type    = quad::QuadratureType::GaussLegendre,
                                                      .use_odd_even = false};
    constexpr auto sum_fact_params_oe = SumFactParams{.basis_order  = EO,
                                                      .quad_order   = QO,
                                                      .basis_type   = basis::BasisType::Lagrange,
                                                      .quad_type    = quad::QuadratureType::GaussLegendre,
                                                      .use_odd_even = true};

    SECTION("Sweep back interp")
    {
        using x_t       = Eigen::Matrix< val_t, sum_fact_params_sf.n_bases1d(), size >;
        using y_t       = Eigen::Matrix< val_t, size, sum_fact_params_sf.n_qps1d() >;
        const auto x    = x_t::Random().eval();
        auto       y_sf = y_t{};
        auto       y_oe = y_t{};
        algsys::detail::sumFactSweepBackInterp< sum_fact_params_sf >(x, y_sf);
        algsys::detail::sumFactSweepBackInterp< sum_fact_params_oe >(x, y_oe);
        CHECK((y_sf - y_oe).norm() < 1e-8);
    }

    SECTION("Sweep back der")
    {
        using x_t       = Eigen::Matrix< val_t, sum_fact_params_sf.n_bases1d(), size >;
        using y_t       = Eigen::Matrix< val_t, size, sum_fact_params_sf.n_qps1d() >;
        const auto x    = x_t::Random().eval();
        auto       y_sf = y_t{};
        auto       y_oe = y_t{};
        algsys::detail::sumFactSweepBackDer< sum_fact_params_sf >(x, y_sf);
        algsys::detail::sumFactSweepBackDer< sum_fact_params_oe >(x, y_oe);
        CHECK((y_sf - y_oe).norm() < 1e-8);
    }

    SECTION("Sweep forward interp assign")
    {
        using x_t       = Eigen::Matrix< val_t, sum_fact_params_sf.n_qps1d(), size >;
        using y_t       = Eigen::Matrix< val_t, size, sum_fact_params_sf.n_bases1d() >;
        const auto x    = x_t::Random().eval();
        auto       y_sf = y_t{};
        auto       y_oe = y_t{};
        algsys::detail::sumFactSweepForwardInterpAssign< sum_fact_params_sf >(x, y_sf);
        algsys::detail::sumFactSweepForwardInterpAssign< sum_fact_params_oe >(x, y_oe);
        CHECK((y_sf - y_oe).norm() < 1e-8);
    }

    SECTION("Sweep forward der accumulate")
    {
        using x_t       = Eigen::Matrix< val_t, sum_fact_params_sf.n_qps1d(), size >;
        using y_t       = Eigen::Matrix< val_t, size, sum_fact_params_sf.n_bases1d() >;
        const auto x    = x_t::Random().eval();
        auto       y_sf = y_t{};
        auto       y_oe = y_t{};
        y_sf.setRandom();
        y_oe = y_sf;
        algsys::detail::sumFactSweepForwardDerAccumulate< sum_fact_params_sf >(x, y_sf);
        algsys::detail::sumFactSweepForwardDerAccumulate< sum_fact_params_oe >(x, y_oe);
        CHECK((y_sf - y_oe).norm() < 1e-8);
    }
}
