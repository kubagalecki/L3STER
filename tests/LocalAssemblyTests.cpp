#include "LocalOperatorCommon.hpp"

TEST_CASE("Local system assembly", "[local_asm]")
{
    // Solve problems using a 1 element discretization and compare with known results
    SECTION("Diffusion 2D")
    {
        constexpr auto element = makeQuadElement();
        constexpr auto ET      = element.type;
        constexpr auto EO      = element.order;
        constexpr auto nat_dim = element.native_dim;
        constexpr auto params  = KernelParams{.dimension = 2, .n_equations = 4, .n_unknowns = 3, .n_rhs = nat_dim};
        const auto     phi     = makeSolution< ET, EO, params >(element);
        auto [A, b]            = assembleDiffusionProblem2D< params >(element);
        applyDirichletBCs< ET, EO, params >(A, b, phi);
        const auto x = std::decay_t< decltype(b) >{A.llt().solve(b)};
        for (el_locind_t node = 0; node != element.nodes.size(); ++node)
        {
            const auto dof = node * params.n_unknowns;
            for (Eigen::Index rhs = 0; rhs != params.n_rhs; ++rhs)
                CHECK(x(dof, rhs) == Approx{phi(node, rhs)}.epsilon(1e-6));
        }
    }

    SECTION("Diffusion 3D")
    {
        constexpr auto element = makeHexElement();
        constexpr auto ET      = element.type;
        constexpr auto EO      = element.order;
        constexpr auto nat_dim = element.native_dim;
        constexpr auto params  = KernelParams{.dimension = 3, .n_equations = 7, .n_unknowns = 4, .n_rhs = nat_dim};
        const auto     phi     = makeSolution< ET, EO, params >(element);
        auto [A, b]            = assembleDiffusionProblem3D< params >(element);
        applyDirichletBCs< ET, EO, params >(A, b, phi);
        const auto x = std::decay_t< decltype(b) >{A.llt().solve(b)};
        for (el_locind_t node = 0; node != element.nodes.size(); ++node)
        {
            const auto dof = node * params.n_unknowns;
            for (Eigen::Index rhs = 0; rhs != params.n_rhs; ++rhs)
                CHECK(x(dof, rhs) == Approx{phi(node, rhs)}.epsilon(1e-6));
        }
    }
}
