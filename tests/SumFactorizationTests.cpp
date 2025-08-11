#include "LocalOperatorCommon.hpp"

TEST_CASE("Sum-factorized evaluation", "[local_asm]")
{
    // Compare results between the local element approach and the sum-factorization technique
    SECTION("Diffusion 2D")
    {
        constexpr auto element = makeQuadElement();
        constexpr auto ET      = element.type;
        constexpr auto EO      = element.order;
        constexpr auto params =
            KernelParams{.dimension = 2, .n_equations = 4, .n_unknowns = 3, .n_fields = 1, .n_rhs = 2};

        auto dom_map = mesh::MeshPartition< EO >::domain_map_t{};
        mesh::pushToDomain(dom_map[0], element);
        const auto global_mesh   = mesh::MeshPartition< EO >{std::move(dom_map), {}};
        const auto local_element = mesh::LocalElementView{element, global_mesh, {}};
        const auto sol_man       = makeRandomlyFilledSolutiondManager(global_mesh, 1);

        auto x = Operand< ET, EO, params >{};
        x.setRandom();

        const auto y_local_element = evalDiffusionOperator2DVar< params >(local_element, x, sol_man);
        const auto y_sum_fact      = evalDiffusionVarOperatorSumFact< params >(local_element, x, sol_man);

        constexpr auto eps = 1e-8;
        CHECK((y_local_element - y_sum_fact).norm() < eps);
    }

    SECTION("Diffusion 3D")
    {
        constexpr auto element = makeHexElement();
        constexpr auto ET      = element.type;
        constexpr auto EO      = element.order;
        constexpr auto params =
            KernelParams{.dimension = 3, .n_equations = 7, .n_unknowns = 4, .n_fields = 1, .n_rhs = 2};

        auto dom_map = mesh::MeshPartition< EO >::domain_map_t{};
        mesh::pushToDomain(dom_map[0], element);
        const auto global_mesh   = mesh::MeshPartition< EO >{std::move(dom_map), {}};
        const auto local_element = mesh::LocalElementView{element, global_mesh, {}};
        const auto sol_man       = makeRandomlyFilledSolutiondManager(global_mesh, 1);

        auto x = Operand< ET, EO, params >{};
        x.setRandom();

        const auto y_local_element = evalDiffusionOperator3DVar< params >(local_element, x, sol_man);
        const auto y_sum_fact      = evalDiffusionVarOperatorSumFact< params >(local_element, x, sol_man);

        constexpr auto eps = 1e-8;
        CHECK((y_local_element - y_sum_fact).norm() < eps);
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
