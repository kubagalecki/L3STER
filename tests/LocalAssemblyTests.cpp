#include "catch2/catch.hpp"

#include "l3ster/algsys/AssembleLocalSystem.hpp"
#include "l3ster/algsys/SumFactorization.hpp"
#include "l3ster/basisfun/ReferenceElementBasisAtQuadrature.hpp"
#include "l3ster/mesh/NodePhysicalLocation.hpp"

using namespace lstr;
using namespace lstr::algsys;

inline constexpr auto diffusion_kernel_2D = [](const auto&, auto& out) noexcept {
    auto& [operators, rhs] = out;
    auto& [A0, A1, A2]     = operators;

    constexpr double lambda = 1.;

    A0(1, 1) = -1.;
    A0(2, 2) = -1.;

    A1(0, 1) = -lambda;
    A1(1, 0) = 1.;
    A1(3, 2) = 1.;

    A2(0, 2) = -lambda;
    A2(2, 0) = 1.;
    A2(3, 1) = -1.;
};

inline constexpr auto diffusion_kernel_3D = [](const auto&, auto& out) noexcept {
    auto& [operators, rhs] = out;
    auto& [A0, Ax, Ay, Az] = operators;

    constexpr double lambda = 1.;

    // -k * div q = s
    Ax(0, 1) = -lambda;
    Ay(0, 2) = -lambda;
    Az(0, 3) = -lambda;

    // grad T = q
    A0(1, 1) = -1.;
    Ax(1, 0) = 1.;
    A0(2, 2) = -1.;
    Ay(2, 0) = 1.;
    A0(3, 3) = -1.;
    Az(3, 0) = 1.;

    // curl q = 0
    Ay(4, 3) = 1.;
    Az(4, 2) = -1.;
    Ax(5, 3) = -1.;
    Az(5, 1) = 1.;
    Ax(6, 2) = 1.;
    Ay(6, 1) = -1.;
};

static constexpr auto makeQuadElement()
{
    constexpr auto   ET = mesh::ElementType::Quad;
    constexpr el_o_t EO = 4;

    auto           el_nodes     = mesh::Element< ET, EO >::node_array_t{};
    constexpr auto invalid_node = std::numeric_limits< n_id_t >::max();
    el_nodes.fill(invalid_node);
    n_id_t n = 0;
    for (const auto& bnd_node_ind : mesh::ElementTraits< mesh::Element< ET, EO > >::boundary_node_inds)
        el_nodes[bnd_node_ind] = n++;
    for (const auto& int_node_ind : mesh::ElementTraits< mesh::Element< ET, EO > >::internal_node_inds)
        el_nodes[int_node_ind] = n++;
    const auto data =
        mesh::ElementData< ET, EO >{{Point{1., 1., 0.}, Point{2., 1., 0.}, Point{1., 3., 0.}, Point{3., 4., 0.}}};

    return mesh::Element{el_nodes, data, 0};
}

static constexpr auto makeHexElement()
{
    constexpr auto   ET = mesh::ElementType::Hex;
    constexpr el_o_t EO = 4;

    auto           el_nodes     = mesh::Element< ET, EO >::node_array_t{};
    constexpr auto invalid_node = std::numeric_limits< n_id_t >::max();
    el_nodes.fill(invalid_node);
    n_id_t n = 0;
    for (const auto& bnd_node_ind : mesh::ElementTraits< mesh::Element< ET, EO > >::boundary_node_inds)
        el_nodes[bnd_node_ind] = n++;
    for (const auto& int_node_ind : mesh::ElementTraits< mesh::Element< ET, EO > >::internal_node_inds)
        el_nodes[int_node_ind] = n++;
    const auto data = mesh::ElementData< ET, EO >{{Point{1., 1., 0.},
                                                   Point{2., 1., 0.},
                                                   Point{1., 3., 0.},
                                                   Point{3., 4., 0.},
                                                   Point{1., 1., 1.},
                                                   Point{2., 1., 1.},
                                                   Point{1., 3., 1.},
                                                   Point{3., 4., 1.}}};

    return mesh::Element{el_nodes, data, 0};
}

inline constexpr auto asm_opts = AssemblyOptions{.value_order = 2}; // over-integrate to trigger local sys buf overflow

template < mesh::ElementType ET, el_o_t EO >
static auto getReferenceBasis() -> const auto&
{
    return basis::getReferenceBasisAtDomainQuadrature< asm_opts.basis_type,
                                                       ET,
                                                       EO,
                                                       asm_opts.quad_type,
                                                       2 * asm_opts.order(EO) >();
}

template < KernelParams params, el_o_t EO >
static auto assembleDiffusionProblem2D(const mesh::Element< mesh::ElementType::Quad, EO >& element)
{
    constexpr auto ET         = mesh::ElementType::Quad;
    constexpr auto kernel     = wrapDomainEquationKernel< params >(diffusion_kernel_2D);
    const auto&    basis_at_q = getReferenceBasis< ET, EO >();
    return assembleLocalSystem(kernel, element, {}, basis_at_q, 0.);
}

template < KernelParams params, el_o_t EO >
static auto evalDiffusionOperator2D(const mesh::LocalElementView< mesh::ElementType::Quad, EO >& element,
                                    const Operand< mesh::ElementType::Quad, EO, params >&        x)
{
    constexpr auto ET         = mesh::ElementType::Quad;
    constexpr auto kernel     = wrapDomainEquationKernel< params >(diffusion_kernel_2D);
    const auto&    basis_at_q = getReferenceBasis< ET, EO >();
    return evaluateLocalOperator(kernel, element, {}, basis_at_q, 0., x);
}

template < KernelParams params, el_o_t EO >
static auto evalDiffusionOperator3D(const mesh::LocalElementView< mesh::ElementType::Hex, EO >& element,
                                    const Operand< mesh::ElementType::Hex, EO, params >&        x)
{
    constexpr auto ET         = mesh::ElementType::Hex;
    constexpr auto kernel     = wrapDomainEquationKernel< params >(diffusion_kernel_3D);
    const auto&    basis_at_q = getReferenceBasis< ET, EO >();
    return evaluateLocalOperator(kernel, element, {}, basis_at_q, 0., x);
}

template < KernelParams params, mesh::ElementType ET, el_o_t EO >
static auto evalDiffusionOperatorSumFact(const mesh::LocalElementView< ET, EO >& element,
                                         const Operand< ET, EO, params >&        x)
{
    constexpr auto            kernel  = std::invoke([] {
        if constexpr (ET == mesh::ElementType::Quad)
            return wrapDomainEquationKernel< params >(diffusion_kernel_2D);
        else
            return wrapDomainEquationKernel< params >(diffusion_kernel_3D);
    });
    constexpr auto            n_nodes = mesh::Element< ET, EO >::n_nodes;
    constexpr Eigen::Index    nukn    = params.n_unknowns;
    Operand< ET, EO, params > y;
    const auto                x_fill = [&x](std::span< val_t > to_fill, size_t stride) {
        util::throwingAssert(stride == params.n_unknowns * params.n_rhs);
        using map_t = Eigen::Map< Eigen::Matrix< val_t, n_nodes, params.n_unknowns * params.n_rhs > >;
        auto to_fill_map = map_t{to_fill.data()};
        for (Eigen::Index i = 0; i != n_nodes; ++i)
            to_fill_map.row(i) = x.template middleRows< nukn >(i * nukn).reshaped().transpose();
    };
    const auto y_fill = [&y](std::span< val_t > result, size_t stride) {
        util::throwingAssert(stride == params.n_unknowns * params.n_rhs);
        using map_t = Eigen::Map< util::eigen::RowMajorMatrix< val_t, n_nodes, params.n_unknowns * params.n_rhs > >;
        const auto result_map = map_t{result.data()};
        for (Eigen::Index i = 0; i != n_nodes; ++i)
            y.template middleRows< nukn >(i * nukn).reshaped().transpose() = result_map.row(i);
    };
    constexpr auto asm_opts_ctwrpr = util::ConstexprValue< asm_opts >{};
    evalLocalOperatorSumFact(kernel, element, x_fill, {}, y_fill, asm_opts_ctwrpr);
    return y;
}

template < KernelParams params, mesh::ElementType ET, el_o_t EO >
static auto initDiffusionOperator2D(const mesh::LocalElementView< ET, EO >& element,
                                    const DirichletInds< ET, EO, params >&  dirichlet_inds = {},
                                    const DirichletVals< ET, EO, params >&  dirichlet_vals = {})
{
    constexpr auto kernel     = wrapDomainEquationKernel< params >(diffusion_kernel_2D);
    const auto&    basis_at_q = getReferenceBasis< ET, EO >();
    return precomputeOperatorDiagonalAndRhs(kernel, element, {}, basis_at_q, 0., dirichlet_inds, dirichlet_vals);
}

template < mesh::ElementType ET, el_o_t EO, KernelParams params >
static void applyDirichletBCs(auto& A, auto& b, const auto& phi)
{
    const auto& bnd_node_inds = mesh::ElementTraits< mesh::Element< ET, EO > >::boundary_node_inds;
    auto        bc_dofs       = std::array< Eigen::Index, bnd_node_inds.size() >{};
    for (auto&& [dof, node] : std::views::zip(bc_dofs, bnd_node_inds))
        dof = node * params.n_unknowns;
    b -= A(Eigen::all, bc_dofs) * phi(bnd_node_inds, Eigen::all);
    b(bc_dofs, Eigen::all) = phi(bnd_node_inds, Eigen::all);
    for (auto i : bc_dofs)
    {
        A.row(i).setZero();
        A.col(i).setZero();
        A(i, i) = 1.;
    }
}

template < mesh::ElementType ET, el_o_t EO, KernelParams params >
static auto makeSolution(const mesh::Element< ET, EO >& element)
{
    // Analytical solution - phi(x) = x
    auto retval = Eigen::Matrix< val_t, mesh::Element< ET, EO >::n_nodes, params.n_rhs >{};
    for (el_locind_t n = 0; n != element.nodes.size(); ++n)
    {
        const auto location = nodePhysicalLocation(element, n);
        retval(n, 0)        = location.x();
        retval(n, 1)        = location.y();
    }
    return retval;
}

TEST_CASE("Local system assembly", "[local_asm]")
{
    // Solve problems using a 1 element discretization and compare with known results
    SECTION("Diffusion 2D")
    {
        constexpr auto element = makeQuadElement();
        constexpr auto ET      = element.type;
        constexpr auto EO      = element.order;
        constexpr auto params  = KernelParams{.dimension = 2, .n_equations = 4, .n_unknowns = 3, .n_rhs = 2};
        const auto     phi     = makeSolution< ET, EO, params >(element);
        auto [A, b]            = assembleDiffusionProblem2D< params >(element);
        applyDirichletBCs< ET, EO, params >(A, b, phi);
        const auto x = std::decay_t< decltype(b) >{A.llt().solve(b)};
        for (el_locind_t node = 0; node != element.nodes.size(); ++node)
        {
            const auto dof = node * params.n_unknowns;
            for (Eigen::Index rhs = 0; rhs != params.n_rhs; ++rhs)
                CHECK(x(dof, rhs) == Approx{phi(node, rhs)}.epsilon(1.e-6));
        }
    }
}

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
}

TEST_CASE("Sum-factorization", "[local_asm]")
{
    // Compare results between the local element approach and the sum-factorization technique
    SECTION("Diffusion 2D")
    {
        constexpr auto element = makeQuadElement();
        constexpr auto ET      = element.type;
        constexpr auto EO      = element.order;
        constexpr auto params  = KernelParams{.dimension = 2, .n_equations = 4, .n_unknowns = 3, .n_rhs = 2};

        auto dom_map = mesh::MeshPartition< EO >::domain_map_t{};
        mesh::pushToDomain(dom_map[0], element);
        const auto global_mesh   = mesh::MeshPartition< EO >{std::move(dom_map), {}};
        const auto local_element = mesh::LocalElementView{element, global_mesh, {}};

        auto x = Operand< ET, EO, params >{};
        x.setRandom();

        const auto y_local_element = evalDiffusionOperator2D< params >(local_element, x);
        const auto y_sum_fact      = evalDiffusionOperatorSumFact< params >(local_element, x);

        constexpr auto eps = 1e-8;
        CHECK((y_local_element - y_sum_fact).norm() < eps);
    }

    SECTION("Diffusion 3D")
    {
        constexpr auto element = makeHexElement();
        constexpr auto ET      = element.type;
        constexpr auto EO      = element.order;
        constexpr auto params  = KernelParams{.dimension = 3, .n_equations = 7, .n_unknowns = 4, .n_rhs = 2};

        auto dom_map = mesh::MeshPartition< EO >::domain_map_t{};
        mesh::pushToDomain(dom_map[0], element);
        const auto global_mesh   = mesh::MeshPartition< EO >{std::move(dom_map), {}};
        const auto local_element = mesh::LocalElementView{element, global_mesh, {}};

        auto x = Operand< ET, EO, params >{};
        x.setRandom();

        const auto y_local_element = evalDiffusionOperator3D< params >(local_element, x);
        const auto y_sum_fact      = evalDiffusionOperatorSumFact< params >(local_element, x);

        constexpr auto eps = 1e-8;
        CHECK((y_local_element - y_sum_fact).norm() < eps);
    }
}
