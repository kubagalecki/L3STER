#ifndef L3STER_TESTS_LOCALOPERATORCOMMON_HPP
#define L3STER_TESTS_LOCALOPERATORCOMMON_HPP

#include "catch2/catch.hpp"

#include "l3ster/algsys/AssembleLocalSystem.hpp"
#include "l3ster/algsys/SumFactorization.hpp"
#include "l3ster/basisfun/ReferenceElementBasisAtQuadrature.hpp"
#include "l3ster/mesh/NodePhysicalLocation.hpp"
#include "l3ster/post/SolutionManager.hpp"

#include "Kernels.hpp"

using namespace lstr;
using namespace lstr::algsys;

constexpr auto makeQuadElement()
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

constexpr auto makeHexElement()
{
    constexpr auto   ET = mesh::ElementType::Hex;
    constexpr el_o_t EO = 3;

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
                                                   Point{2., 1., 1.5},
                                                   Point{1., 3., 2.},
                                                   Point{3., 4., 3.5}}};

    return mesh::Element{el_nodes, data, 0};
}

inline constexpr auto asm_opts = AssemblyOptions{.value_order = 2}; // over-integrate to trigger local sys buf overflow

template < mesh::ElementType ET, el_o_t EO >
auto getReferenceBasis() -> const auto&
{
    return basis::getReferenceBasisAtDomainQuadrature< asm_opts.basis_type,
                                                       ET,
                                                       EO,
                                                       asm_opts.quad_type,
                                                       2 * asm_opts.order(EO) >();
}

template < KernelParams params, el_o_t EO >
auto assembleDiffusionProblem2D(const mesh::Element< mesh::ElementType::Quad, EO >& element)
{
    constexpr auto ET         = mesh::ElementType::Quad;
    constexpr auto kernel     = wrapDomainEquationKernel< params >(diffusion_kernel_2D);
    const auto&    basis_at_q = getReferenceBasis< ET, EO >();
    return assembleLocalSystem(kernel, element, {}, basis_at_q, 0.);
}

template < KernelParams params, el_o_t EO >
auto assembleDiffusionProblem3D(const mesh::Element< mesh::ElementType::Hex, EO >& element)
{
    constexpr auto ET         = mesh::ElementType::Hex;
    constexpr auto kernel     = wrapDomainEquationKernel< params >(diffusion_kernel_3D);
    const auto&    basis_at_q = getReferenceBasis< ET, EO >();
    return assembleLocalSystem(kernel, element, {}, basis_at_q, 0.);
}

template < KernelParams params, el_o_t EO >
auto evalDiffusionOperator2D(const mesh::LocalElementView< mesh::ElementType::Quad, EO >& element,
                             const Operand< mesh::ElementType::Quad, EO, params >&        x)
{
    constexpr auto ET         = mesh::ElementType::Quad;
    constexpr auto kernel     = wrapDomainEquationKernel< params >(diffusion_kernel_2D);
    const auto&    basis_at_q = getReferenceBasis< ET, EO >();
    return evaluateLocalOperator(kernel, element, {}, basis_at_q, 0., x);
}

template < KernelParams params, el_o_t EO >
auto evalDiffusionOperator3D(const mesh::LocalElementView< mesh::ElementType::Hex, EO >& element,
                             const Operand< mesh::ElementType::Hex, EO, params >&        x)
{
    constexpr auto ET         = mesh::ElementType::Hex;
    constexpr auto kernel     = wrapDomainEquationKernel< params >(diffusion_kernel_3D);
    const auto&    basis_at_q = getReferenceBasis< ET, EO >();
    return evaluateLocalOperator(kernel, element, {}, basis_at_q, 0., x);
}

template < KernelParams params, el_o_t EO >
auto evalDiffusionOperator2DVar(const mesh::LocalElementView< mesh::ElementType::Quad, EO >& element,
                                const Operand< mesh::ElementType::Quad, EO, params >&        x,
                                const SolutionManager&                                       sol_man)
{
    constexpr auto ET         = mesh::ElementType::Quad;
    constexpr auto kernel     = wrapDomainEquationKernel< params >(diffusion_kernel_2D_var);
    const auto&    basis_at_q = getReferenceBasis< ET, EO >();
    const auto     node_vals  = sol_man.getFieldAccess(std::array{0}).getLocallyIndexed(element.getLocalNodes());
    return evaluateLocalOperator(kernel, element, node_vals, basis_at_q, 0., x);
}

template < KernelParams params, el_o_t EO >
auto evalDiffusionOperator3DVar(const mesh::LocalElementView< mesh::ElementType::Hex, EO >& element,
                                const Operand< mesh::ElementType::Hex, EO, params >&        x,
                                const SolutionManager&                                      sol_man)
{
    constexpr auto ET         = mesh::ElementType::Hex;
    constexpr auto kernel     = wrapDomainEquationKernel< params >(diffusion_kernel_3D_var);
    const auto&    basis_at_q = getReferenceBasis< ET, EO >();
    const auto     node_vals  = sol_man.getFieldAccess(std::array{0}).getLocallyIndexed(element.getLocalNodes());
    return evaluateLocalOperator(kernel, element, node_vals, basis_at_q, 0., x);
}

template < KernelParams params, mesh::ElementType ET, el_o_t EO >
auto evalDiffusionVarOperatorSumFact(const mesh::LocalElementView< ET, EO >& element,
                                     const Operand< ET, EO, params >&        x,
                                     const SolutionManager&                  sol_man)
{
    constexpr auto            kernel  = std::invoke([] {
        if constexpr (ET == mesh::ElementType::Quad)
            return wrapDomainEquationKernel< params >(diffusion_kernel_2D_var);
        else
            return wrapDomainEquationKernel< params >(diffusion_kernel_3D_var);
    });
    constexpr auto            n_nodes = mesh::Element< ET, EO >::n_nodes;
    constexpr Eigen::Index    nukn    = params.n_unknowns;
    Operand< ET, EO, params > y;
    const auto                x_fill = [&x](std::span< val_t > to_fill) {
        using map_t = Eigen::Map< Eigen::Matrix< val_t, n_nodes, params.n_unknowns * params.n_rhs > >;
        auto to_fill_map = map_t{to_fill.data()};
        for (Eigen::Index i = 0; i != n_nodes; ++i)
            to_fill_map.row(i) = x.template middleRows< nukn >(i * nukn).reshaped().transpose();
    };
    const auto y_fill = [&y](std::span< val_t > result) {
        using map_t = Eigen::Map< util::eigen::RowMajorMatrix< val_t, n_nodes, params.n_unknowns * params.n_rhs > >;
        const auto result_map = map_t{result.data()};
        for (Eigen::Index i = 0; i != n_nodes; ++i)
            y.template middleRows< nukn >(i * nukn).reshaped().transpose() = result_map.row(i);
    };
    constexpr auto asm_opts_ctwrpr = util::ConstexprValue< asm_opts >{};
    evalLocalOperatorSumFact(kernel, element, x_fill, sol_man.getFieldAccess(std::array{0}), y_fill, asm_opts_ctwrpr);
    return y;
}

template < KernelParams params, mesh::ElementType ET, el_o_t EO >
auto initDiffusionOperator2D(const mesh::LocalElementView< ET, EO >& element,
                             const DirichletInds< ET, EO, params >&  dirichlet_inds = {},
                             const DirichletVals< ET, EO, params >&  dirichlet_vals = {})
{
    constexpr auto kernel     = wrapDomainEquationKernel< params >(diffusion_kernel_2D);
    const auto&    basis_at_q = getReferenceBasis< ET, EO >();
    return precomputeOperatorDiagonalAndRhs(kernel, element, {}, basis_at_q, 0., dirichlet_inds, dirichlet_vals);
}

template < KernelParams params, mesh::ElementType ET, el_o_t EO >
auto initDiffusionOperator3D(const mesh::LocalElementView< ET, EO >& element,
                             const DirichletInds< ET, EO, params >&  dirichlet_inds = {},
                             const DirichletVals< ET, EO, params >&  dirichlet_vals = {})
{
    constexpr auto kernel     = wrapDomainEquationKernel< params >(diffusion_kernel_3D);
    const auto&    basis_at_q = getReferenceBasis< ET, EO >();
    return precomputeOperatorDiagonalAndRhs(kernel, element, {}, basis_at_q, 0., dirichlet_inds, dirichlet_vals);
}

template < mesh::ElementType ET, el_o_t EO, KernelParams params >
void applyDirichletBCs(auto& A, auto& b, const auto& phi)
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
auto makeSolution(const mesh::Element< ET, EO >& element)
{
    // Analytical solution - phi(x) = x
    auto retval = Eigen::Matrix< val_t, mesh::Element< ET, EO >::n_nodes, params.n_rhs >{};
    for (el_locind_t n = 0; n != element.nodes.size(); ++n)
    {
        const auto location = nodePhysicalLocation(element, n);
        for (size_t d = 0; d != mesh::Element< ET, EO >::native_dim; ++d)
            retval(n, d) = location[d];
    }
    return retval;
}

template < el_o_t... EO >
auto makeRandomlyFilledSolutiondManager(const mesh::MeshPartition< EO... >& partition, size_t n_fields)
{
    auto retval   = SolutionManager{partition, n_fields};
    auto raw_view = retval.getRawView();
    auto prng     = std::mt19937_64{};
    auto dist     = std::uniform_real_distribution< val_t >{-1., 1.};
    for (size_t row = 0; row != raw_view.extent(0); ++row)
        for (size_t col = 0; col != raw_view.extent(1); ++col)
            raw_view(row, col) = dist(prng);
    return retval;
}

#endif // L3STER_TESTS_LOCALOPERATORCOMMON_HPP
