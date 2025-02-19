#ifndef L3STER_ALGSYS_MATRIXFREESYSTEM_HPP
#define L3STER_ALGSYS_MATRIXFREESYSTEM_HPP

#include "l3ster/algsys/ComputeValuesAtNodes.hpp"
#include "l3ster/algsys/EvaluateLocalOperator.hpp"
#include "l3ster/algsys/SparsityGraph.hpp"
#include "l3ster/basisfun/ReferenceElementBasisAtQuadrature.hpp"
#include "l3ster/bcs/LocalDirichletBC.hpp"
#include "l3ster/comm/ImportExport.hpp"
#include "l3ster/mesh/SplitMesh.hpp"
#include "l3ster/post/FieldAccess.hpp"
#include "l3ster/solve/SolverInterface.hpp"
#include "l3ster/util/GlobalResource.hpp"
#include "l3ster/util/TypeErasedOverload.hpp"

namespace lstr::algsys
{
template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
class MatrixFreeSystem
{
public:
    friend class Operator;
    class Operator final : public tpetra_operator_t
    {
    public:
        Operator(const MatrixFreeSystem* system) : m_system{system} {}

        auto getDomainMap() const -> Teuchos::RCP< const tpetra_map_t > final { return m_system->getOperatorMap(); }
        auto getRangeMap() const -> Teuchos::RCP< const tpetra_map_t > final { return m_system->getOperatorMap(); }
        bool hasTransposeApply() const final { return true; }
        bool hasDiagonal() const final { return true; }
        void getLocalDiagCopy(tpetra_vector_t& diag) const final { m_system->copyDiagImpl(diag); }
        void apply(const tpetra_multivector_t& x,
                   tpetra_multivector_t&       y,
                   Teuchos::ETransp, // operator is symmetric, we can ignore the mode
                   val_t alpha,
                   val_t beta) const override
        {
            m_system->applyImpl(x, y, alpha, beta);
        }

    private:
        const MatrixFreeSystem* m_system;
    };

    inline MatrixFreeSystem(std::shared_ptr< const MpiComm >                          comm,
                            std::shared_ptr< const mesh::MeshPartition< orders... > > mesh,
                            const ProblemDefinition< max_dofs_per_node >&             problem_def,
                            const BCDefinition< max_dofs_per_node >&                  bc_def);

    [[nodiscard]] inline auto getOperator() const -> Teuchos::RCP< const tpetra_operator_t >;
    [[nodiscard]] inline auto getRhs() const -> Teuchos::RCP< const tpetra_multivector_t >;
    [[nodiscard]] inline auto getSolution() const -> Teuchos::RCP< tpetra_multivector_t >;

    inline void beginAssembly();
    inline void endAssembly();
    template < EquationKernel_c         Kernel,
               ArrayOf_c< size_t > auto field_inds = util::makeIotaArray< size_t, max_dofs_per_node >(),
               size_t                   n_fields   = 0,
               AssemblyOptions          asm_opts   = AssemblyOptions{} >
    void assembleProblem(const Kernel&                        kernel,
                         const util::ArrayOwner< d_id_t >&    domain_ids,
                         const post::FieldAccess< n_fields >& field_access      = {},
                         util::ConstexprValue< field_inds >   field_inds_ctwrpr = {},
                         util::ConstexprValue< asm_opts >     asm_options       = {},
                         val_t                                time              = 0.)
        requires(Kernel::parameters.n_rhs == n_rhs);

    template < ResidualKernel_c Kernel, std::integral dofind_t = size_t, size_t n_fields = 0 >
    void setDirichletBCValues(const Kernel&                                                 kernel,
                              const util::ArrayOwner< d_id_t >&                             domain_ids,
                              const std::array< dofind_t, Kernel::parameters.n_equations >& dof_inds =
                                  util::makeIotaArray< dofind_t, Kernel::parameters.n_equations >(),
                              const post::FieldAccess< n_fields >& field_access = {},
                              val_t                                time         = 0.);
    template < size_t n_vals, std::integral dofind_t = size_t >
    void
    setDirichletBCValues(const std::array< val_t, n_vals >&    values,
                         const util::ArrayOwner< d_id_t >&     domain_ids,
                         const std::array< dofind_t, n_vals >& dof_inds = util::makeIotaArray< dofind_t, n_vals >())
        requires(n_rhs == 1);

    template < solvers::IterativeSolver_c Solver >
    IterSolveResult solve(Solver& solver) const;

    inline void updateSolution(const util::ArrayOwner< size_t >& sol_inds,
                               SolutionManager&                  sol_man,
                               const util::ArrayOwner< size_t >& sol_man_inds);

    inline void describe(std::ostream& out = std::cout) const;

    template < ResidualKernel_c Kernel, std::integral dofind_t = size_t, size_t n_fields = 0 >
    void setValues(const Teuchos::RCP< tpetra_multivector_t >&                   vector,
                   const Kernel&                                                 kernel,
                   const util::ArrayOwner< d_id_t >&                             domain_ids,
                   const std::array< dofind_t, Kernel::parameters.n_equations >& dof_inds,
                   const post::FieldAccess< n_fields >&                          field_access = {},
                   val_t                                                         time         = 0.);

private:
    using view_t       = decltype(std::declval< tpetra_multivector_t >().getLocalViewHost(Tpetra::Access::ReadWrite));
    using const_view_t = decltype(std::declval< tpetra_multivector_t >().getLocalViewHost(Tpetra::Access::ReadOnly));

    template < mesh::ElementType ET, el_o_t EO, typename... Args >
    using ElementCallable = void(const mesh::LocalElementView< ET, EO >&, Args...);
    template < mesh::ElementType ET, el_o_t EO, typename... Args >
    using BoundaryElementCallable = void(const mesh::LocalElementBoundaryView< ET, EO >&, Args...);
    template < mesh::ElementType ET, el_o_t EO >
    using DomainInitializer = ElementCallable< ET, EO >;
    template < mesh::ElementType ET, el_o_t EO >
    using BoundaryInitializer = BoundaryElementCallable< ET, EO >;
    template < mesh::ElementType ET, el_o_t EO >
    using DomainEvaluator = ElementCallable< ET, EO, const const_view_t&, const view_t&, val_t >;
    template < mesh::ElementType ET, el_o_t EO >
    using BoundaryEvaluator = BoundaryElementCallable< ET, EO, const const_view_t&, const view_t&, val_t >;

    using DomainInitializerOverload = mesh::
        parametrize_type_over_element_types_and_orders_t< util::TypeErasedOverload, DomainInitializer, orders... >;
    using BoundaryInitializerOverload = mesh::
        parametrize_type_over_element_types_and_orders_t< util::TypeErasedOverload, BoundaryInitializer, orders... >;
    using DomainEvaluatorOverload =
        mesh::parametrize_type_over_element_types_and_orders_t< util::TypeErasedOverload, DomainEvaluator, orders... >;
    using BoundaryEvaluatorOverload = mesh::
        parametrize_type_over_element_types_and_orders_t< util::TypeErasedOverload, BoundaryEvaluator, orders... >;

    template < typename Overload >
    struct Kernels
    {
        Overload interior, border;
    };
    using DomainInitKernels   = Kernels< DomainInitializerOverload >;
    using BoundaryInitKernels = Kernels< BoundaryInitializerOverload >;
    using DomainEvalKernels   = Kernels< DomainEvaluatorOverload >;
    using BoundaryEvalKernels = Kernels< BoundaryEvaluatorOverload >;

    struct KernelMaps
    {
        template < typename F >
        void applyToMaps(F&& fun)
        {
            fun(domain_init);
            fun(boundary_init);
            fun(domain_eval);
            fun(boundary_eval);
        }

        std::map< d_id_t, std::vector< DomainInitKernels > >   domain_init;
        std::map< d_id_t, std::vector< BoundaryInitKernels > > boundary_init;
        std::map< d_id_t, std::vector< DomainEvalKernels > >   domain_eval;
        std::map< d_id_t, std::vector< BoundaryEvalKernels > > boundary_eval;
    };

    enum struct State : std::uint8_t
    {
        OpenForAssembly,
        Closed
    };

    inline void applyImpl(const tpetra_multivector_t& x, tpetra_multivector_t& y, val_t alpha, val_t beta) const;
    auto        getOperatorMap() const -> Teuchos::RCP< const tpetra_map_t > { return m_operator_map; }
    inline void copyDiagImpl(tpetra_vector_t& diag) const;
    inline void computeDiagAndRhs();
    inline void assertState(State                expected,
                            std::string_view     err_msg,
                            std::source_location src_loc = std::source_location::current()) const;
    inline void importDirichletBCs();
    inline void initKernelMaps();
    inline void zeroExportBuf() const;

    template < EquationKernel_c Kernel, auto field_inds, size_t n_fields, AssemblyOptions asm_opts >
    void pushKernel(const Kernel&                        kernel,
                    const util::ArrayOwner< d_id_t >&    domain_ids,
                    const post::FieldAccess< n_fields >& field_access,
                    util::ConstexprValue< field_inds >   field_inds_ctwrpr,
                    util::ConstexprValue< asm_opts >,
                    val_t time);
    template < typename AccessGenerator,
               EquationKernel_c Kernel,
               auto             field_inds,
               size_t           n_fields,
               AssemblyOptions  asm_opts >
    auto makeInitKernel(AccessGenerator&&                    rhs_access_generator,
                        const Kernel&                        kernel,
                        const post::FieldAccess< n_fields >& field_access,
                        util::ConstexprValue< field_inds >   field_inds_ctwrpr,
                        util::ConstexprValue< asm_opts >     asm_options,
                        val_t                                time)
        -> std::conditional_t< DomainKernel_c< Kernel >, DomainInitializerOverload, BoundaryInitializerOverload >;
    template < typename XAccessGenerator,
               typename YAccessGenerator,
               EquationKernel_c Kernel,
               auto             field_inds,
               size_t           n_fields,
               AssemblyOptions  asm_opts >
    auto makeEvalKernel(XAccessGenerator&&                   x_access_generator,
                        YAccessGenerator&&                   y_access_generator,
                        const Kernel&                        kernel,
                        const post::FieldAccess< n_fields >& field_access,
                        util::ConstexprValue< field_inds >   field_inds_ctwrpr,
                        util::ConstexprValue< asm_opts >     asm_options,
                        val_t                                time)
        -> std::conditional_t< DomainKernel_c< Kernel >, DomainEvaluatorOverload, BoundaryEvaluatorOverload >;

    std::shared_ptr< const MpiComm >                          m_comm;
    std::shared_ptr< const mesh::MeshPartition< orders... > > m_mesh;
    mesh::LocalMeshView< orders... >                          m_interior_mesh, m_border_mesh;
    Teuchos::RCP< tpetra_multivector_t >                      m_rhs, m_solution;
    view_t                                                    m_rhs_view;
    util::ArrayOwner< val_t >                                 m_diagonal;
    std::unique_ptr< comm::Import< val_t > >                  m_import;
    std::unique_ptr< comm::Export< val_t > >                  m_export;
    view_t                                                    m_import_shared_buf, m_export_shared_buf;
    std::optional< bcs::LocalDirichletBC >                    m_dirichlet_bc;
    view_t                                                    m_dirichlet_values;
    dofs::LocalDofMap< max_dofs_per_node >                    m_node_dof_map;
    KernelMaps                                                m_kernel_maps;
    Teuchos::RCP< const tpetra_map_t >                        m_operator_map;
    State                                                     m_state;
    bool                                                      m_bcs_need_import = false;
};

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
void MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::zeroExportBuf() const
{
    util::tbb::parallelFor(std::views::iota(0uz, n_rhs), [this](size_t rhs) {
        util::tbb::parallelFor(std::views::iota(0uz, m_node_dof_map.getNumSharedDofs()),
                               [this, rhs](size_t i) { m_export_shared_buf(i, rhs) = 0.; });
    });
}

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
template < solvers::IterativeSolver_c Solver >
IterSolveResult MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::solve(Solver& solver) const
{
    L3STER_PROFILE_FUNCTION;
    return solver.solve(getOperator(), m_rhs, m_solution);
}

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
void MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::initKernelMaps()
{
    m_kernel_maps.applyToMaps([&](auto& map) {
        for (d_id_t domain : m_mesh->getDomainIds())
            map[domain].clear();
    });
}

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
void MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::importDirichletBCs()
{
    if (not m_bcs_need_import)
        return;

    const auto num_owned    = m_node_dof_map.getNumOwnedDofs();
    const auto num_all      = m_node_dof_map.getNumTotalDofs();
    const auto owned_range  = std::pair< size_t, size_t >{0, num_owned};
    const auto shared_range = std::pair< size_t, size_t >{num_owned, num_all};
    const auto owned_view   = Kokkos::subview(m_dirichlet_values, owned_range, Kokkos::ALL);
    const auto shared_view  = Kokkos::subview(m_dirichlet_values, shared_range, Kokkos::ALL);
    m_import->setOwned(owned_view);
    m_import->setShared(shared_view);
    m_import->doBlockingImport(*m_comm);
    m_bcs_need_import = false;
}

namespace detail
{
template < size_t num_nodes, size_t max_dofs_per_node, auto field_inds >
auto getDofs(const std::array< n_loc_id_t, num_nodes >&    nodes,
             const dofs::LocalDofMap< max_dofs_per_node >& dof_map,
             util::ConstexprValue< field_inds >) -> std::array< local_dof_t, field_inds.size() * num_nodes >
{
    auto retval = std::array< local_dof_t, field_inds.size() * num_nodes >{};
    for (size_t i = 0; n_loc_id_t n : nodes)
    {
        const auto& node_dofs = dof_map(n);
        for (auto index : field_inds)
            retval[i++] = node_dofs[index];
    }
    return retval;
}

template < size_t num_dofs >
auto getDirichletDofInds(const std::array< local_dof_t, num_dofs >&    dofs,
                         const std::optional< bcs::LocalDirichletBC >& bcs)
    -> util::StaticVector< util::smallest_integral_t< num_dofs >, num_dofs >
{
    using index_t = util::smallest_integral_t< num_dofs >;
    auto retval   = util::StaticVector< index_t, num_dofs >{};
    if (bcs and not bcs->isEmpty())
        for (auto&& [i, dof] : dofs | std::views::enumerate)
            if (bcs->isDirichletDof(dof))
                retval.push_back(static_cast< index_t >(i));
    return retval;
}

template < mesh::ElementType ET, el_o_t EO, KernelParams params, size_t num_dofs, util::KokkosView_c Vals >
auto gatherDirichletVals(
    const std::array< local_dof_t, num_dofs >&                                   dofs,
    const util::StaticVector< util::smallest_integral_t< num_dofs >, num_dofs >& dirichlet_dof_inds,
    const Vals& dirichlet_vals) -> DirichletVals< ET, EO, params >
{
    auto retval = DirichletVals< ET, EO, params >(dirichlet_dof_inds.size(), params.n_rhs);
    for (auto&& [i, dof_ind] : dirichlet_dof_inds | std::views::enumerate)
        for (size_t rhs = 0; rhs != params.n_rhs; ++rhs)
            retval(i, rhs) = dirichlet_vals(dofs[dof_ind], rhs);
    return retval;
}

template < size_t num_rhs, size_t num_dofs, typename RhsLocal, typename DiagLocal, typename RhsGlobal >
void scatterInit(const std::array< local_dof_t, num_dofs >& dofs,
                 const RhsLocal&                            rhs_local,
                 const DiagLocal&                           diag_local,
                 const RhsGlobal&                           rhs_global,
                 std::span< val_t >                         diag_global)
{
    for (auto&& [i, dof] : dofs | std::views::enumerate)
    {
        std::atomic_ref{diag_global[dof]}.fetch_add(diag_local[i]);
        for (local_dof_t rhs = 0; rhs != num_rhs; ++rhs)
            std::atomic_ref{rhs_global(dof, rhs)}.fetch_add(rhs_local(i, rhs));
    }
}

template < local_dof_t num_rhs, size_t num_dofs, typename Access >
auto gather(Access&&                                      from_access,
            const std::array< local_dof_t, num_dofs >&    dofs,
            const std::optional< bcs::LocalDirichletBC >& bcs) -> Eigen::Matrix< val_t, num_dofs, num_rhs >
{
    L3STER_PROFILE_FUNCTION;
    const bool dirichlet_bcs_exist = bcs.has_value() and not bcs->isEmpty();
    auto       retval              = Eigen::Matrix< val_t, num_dofs, num_rhs >{};
    const auto gather_dof          = [&](auto i, local_dof_t dof) {
        for (local_dof_t rhs = 0; rhs != num_rhs; ++rhs)
            retval(i, rhs) = from_access(dof, rhs);
    };
    if (dirichlet_bcs_exist)
    {
        for (auto&& [i, dof] : dofs | std::views::enumerate)
            if (bcs->isDirichletDof(dof))
                retval.row(i).setZero();
            else
                gather_dof(i, dof);
    }
    else
        for (auto&& [i, dof] : dofs | std::views::enumerate)
            gather_dof(i, dof);
    return retval;
}

template < int num_dofs, int num_rhs, typename Access >
void scatter(const Eigen::Matrix< val_t, num_dofs, num_rhs >&                  from,
             Access&&                                                          to_access,
             const std::array< local_dof_t, static_cast< size_t >(num_dofs) >& dofs,
             const std::optional< bcs::LocalDirichletBC >&                     bcs,
             val_t                                                             scale)
{
    L3STER_PROFILE_FUNCTION;
    const bool dirichlet_bcs_exist = bcs.has_value() and not bcs->isEmpty();
    const auto scatter_dof         = [&](auto i, local_dof_t dof) {
        for (local_dof_t rhs = 0; rhs != num_rhs; ++rhs)
            std::atomic_ref{to_access(dof, rhs)}.fetch_add(from(i, rhs) * scale, std::memory_order_relaxed);
    };
    if (dirichlet_bcs_exist)
    {
        for (auto&& [i, dof] : dofs | std::views::enumerate)
            if (not bcs->isDirichletDof(dof))
                scatter_dof(i, dof);
    }
    else
        for (auto&& [i, dof] : dofs | std::views::enumerate)
            scatter_dof(i, dof);
}

template < mesh::ElementType ET, el_o_t EO, typename KernelPtr, typename KernelMap, typename... Args >
void invokeBoundaryKernels(const mesh::LocalElementView< ET, EO >& element,
                           KernelPtr                               overload_ptr,
                           const KernelMap&                        kernel_map,
                           Args&&... args)
{
    for (const auto& [side, boundary] : element.getBoundaries())
    {
        const auto boundary_view = mesh::LocalElementBoundaryView{&element, side};
        for (const auto& boundary_kernels : kernel_map.at(boundary))
            std::invoke(boundary_kernels.*overload_ptr, boundary_view, args...);
    }
}

template < auto              field_inds,
           KernelParams      params,
           size_t            n_fields,
           size_t            max_dofs_per_node,
           mesh::ElementType ET,
           el_o_t            EO >
struct CommonElemData
{
    static_assert(field_inds.size() == params.n_unknowns);
    static constexpr auto n_nodes = mesh::Element< ET, EO >::n_nodes;
    static constexpr auto n_dofs  = params.n_unknowns * n_nodes;

    CommonElemData(const mesh::LocalElementView< ET, EO >&       element,
                   const post::FieldAccess< n_fields >&          field_access,
                   const dofs::LocalDofMap< max_dofs_per_node >& dof_map,
                   const std::optional< bcs::LocalDirichletBC >& dirichlet_bc,
                   const tpetra_multivector_t::host_view_type&   dirichlet_vals)
    {
        const auto& el_nodes = element.getLocalNodes();
        node_values          = field_access.getLocallyIndexed(el_nodes);
        dofs                 = detail::getDofs(el_nodes, dof_map, util::ConstexprValue< field_inds >{});
        dirichlet_dof_inds   = detail::getDirichletDofInds(dofs, dirichlet_bc);
        dirichlet_values     = detail::gatherDirichletVals< ET, EO, params >(dofs, dirichlet_dof_inds, dirichlet_vals);
    }

    util::eigen::RowMajorMatrix< val_t, n_nodes, n_fields >           node_values;
    std::array< local_dof_t, n_dofs >                                 dofs;
    util::StaticVector< util::smallest_integral_t< n_dofs >, n_dofs > dirichlet_dof_inds;
    DirichletVals< ET, EO, params >                                   dirichlet_values;
};
} // namespace detail

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
template < typename AccessGenerator,
           EquationKernel_c Kernel,
           auto             field_inds,
           size_t           n_fields,
           AssemblyOptions  asm_opts >
auto MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::makeInitKernel(
    AccessGenerator&&                    rhs_access_generator,
    const Kernel&                        kernel,
    const post::FieldAccess< n_fields >& field_access,
    util::ConstexprValue< field_inds >,
    util::ConstexprValue< asm_opts >,
    val_t time)
    -> std::conditional_t< DomainKernel_c< Kernel >, DomainInitializerOverload, BoundaryInitializerOverload >
{
    return [=, this](const auto& element) {
        constexpr auto BT         = asm_opts.basis_type;
        constexpr auto QT         = asm_opts.quad_type;
        constexpr auto params     = Kernel::parameters;
        constexpr auto ET         = std::decay_t< decltype(element) >::type;
        constexpr auto EO         = std::decay_t< decltype(element) >::order;
        using CommonData          = detail::CommonElemData< field_inds, params, n_fields, max_dofs_per_node, ET, EO >;
        static constexpr q_o_t QO = 2 * asm_opts.order(EO);
        constexpr bool is_domain  = std::same_as< std::decay_t< decltype(element) >, mesh::LocalElementView< ET, EO > >;
        if constexpr (mesh::Element< ET, EO >::native_dim == params.dimension)
        {
            const auto rhs_access          = rhs_access_generator(m_rhs_view);
            const auto get_reference_basis = [&element] -> const auto& {
                if constexpr (is_domain)
                    return basis::getReferenceBasisAtDomainQuadrature< BT, ET, EO, QT, QO >();
                else
                    return basis::getReferenceBasisAtBoundaryQuadrature< BT, ET, EO, QT, QO >(element.getSide());
            };
            const auto get_data = [&] -> CommonData {
                if constexpr (is_domain)
                    return {element, field_access, m_node_dof_map, m_dirichlet_bc, m_dirichlet_values};
                else
                    return {*element, field_access, m_node_dof_map, m_dirichlet_bc, m_dirichlet_values};
            };
            const auto [node_vals, dofs, dir_dof_inds, dir_vals] = get_data();
            const auto& rbq                                      = get_reference_basis();
            const auto [loc_diag, loc_rhs] =
                precomputeOperatorDiagonalAndRhs(kernel, element, node_vals, rbq, time, dir_dof_inds, dir_vals);
            detail::scatterInit< params.n_rhs >(dofs, loc_rhs, loc_diag, rhs_access, m_diagonal);
        }
    };
}

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
template < typename XAccessGenerator,
           typename YAccessGenerator,
           EquationKernel_c Kernel,
           auto             field_inds,
           size_t           n_fields,
           AssemblyOptions  asm_opts >
auto MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::makeEvalKernel(
    XAccessGenerator&&                   x_access_generator,
    YAccessGenerator&&                   y_access_generator,
    const Kernel&                        kernel,
    const post::FieldAccess< n_fields >& field_access,
    util::ConstexprValue< field_inds >,
    util::ConstexprValue< asm_opts >,
    val_t time) -> std::conditional_t< DomainKernel_c< Kernel >, DomainEvaluatorOverload, BoundaryEvaluatorOverload >
{
    static constexpr auto params = Kernel::parameters;
    return [=, this](const auto& element, const const_view_t& x, const view_t& y, val_t alpha) {
        constexpr auto BT         = asm_opts.basis_type;
        constexpr auto QT         = asm_opts.quad_type;
        constexpr auto ET         = std::decay_t< decltype(element) >::type;
        constexpr auto EO         = std::decay_t< decltype(element) >::order;
        using CommonData          = detail::CommonElemData< field_inds, params, n_fields, max_dofs_per_node, ET, EO >;
        static constexpr q_o_t QO = 2 * asm_opts.order(EO);
        constexpr bool is_domain  = std::same_as< std::decay_t< decltype(element) >, mesh::LocalElementView< ET, EO > >;
        if constexpr (mesh::Element< ET, EO >::native_dim == params.dimension)
        {
            const auto get_reference_basis = [&element] -> const auto& {
                if constexpr (is_domain)
                    return basis::getReferenceBasisAtDomainQuadrature< BT, ET, EO, QT, QO >();
                else
                    return basis::getReferenceBasisAtBoundaryQuadrature< BT, ET, EO, QT, QO >(element.getSide());
            };
            const auto get_data = [&] -> CommonData {
                if constexpr (is_domain)
                    return {element, field_access, m_node_dof_map, m_dirichlet_bc, m_dirichlet_values};
                else
                    return {*element, field_access, m_node_dof_map, m_dirichlet_bc, m_dirichlet_values};
            };
            const auto [node_vals, dofs, dir_dof_inds, dir_vals] = get_data();
            const auto& rbq                                      = get_reference_basis();
            const auto  x_access                                 = x_access_generator(x);
            const auto  y_access                                 = y_access_generator(y);
            const auto  x_local = detail::gather< params.n_rhs >(x_access, dofs, m_dirichlet_bc);
            const auto  y_local = evaluateLocalOperator(kernel, element, node_vals, rbq, time, x_local);
            detail::scatter(y_local, y_access, dofs, m_dirichlet_bc, alpha);
        }
    };
}

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
template < EquationKernel_c Kernel, auto field_inds, size_t n_fields, AssemblyOptions asm_opts >
void MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::pushKernel(
    const Kernel&                        kernel,
    const util::ArrayOwner< d_id_t >&    domain_ids,
    const post::FieldAccess< n_fields >& field_access,
    util::ConstexprValue< field_inds >   field_inds_ctwrpr,
    util::ConstexprValue< asm_opts >     asm_options,
    val_t                                time)
{
    constexpr auto int_gen = [](const auto& view) {
        return InteriorAccessor{view};
    };
    const auto imp_gen = [this](const const_view_t& x) {
        return BorderAccessor{x, const_view_t{m_import_shared_buf}};
    };
    const auto exp_gen = [this](const view_t& y) {
        return BorderAccessor{y, m_export_shared_buf};
    };

    for (d_id_t domain : domain_ids)
    {
        auto iinit = makeInitKernel(int_gen, kernel, field_access, field_inds_ctwrpr, asm_options, time);
        auto binit = makeInitKernel(exp_gen, kernel, field_access, field_inds_ctwrpr, asm_options, time);
        auto ieval = makeEvalKernel(int_gen, int_gen, kernel, field_access, field_inds_ctwrpr, asm_options, time);
        auto beval = makeEvalKernel(imp_gen, exp_gen, kernel, field_access, field_inds_ctwrpr, asm_options, time);

        if constexpr (BoundaryKernel_c< Kernel >)
        {
            auto init_kernels = BoundaryInitKernels{.interior = std::move(iinit), .border = std::move(binit)};
            auto eval_kernels = BoundaryEvalKernels{.interior = std::move(ieval), .border = std::move(beval)};
            m_kernel_maps.boundary_init[domain].push_back(std::move(init_kernels));
            m_kernel_maps.boundary_eval[domain].push_back(std::move(eval_kernels));
        }
        else
        {
            static_assert(DomainKernel_c< Kernel >);
            auto init_kernels = DomainInitKernels{.interior = std::move(iinit), .border = std::move(binit)};
            auto eval_kernels = DomainEvalKernels{.interior = std::move(ieval), .border = std::move(beval)};
            m_kernel_maps.domain_init[domain].push_back(std::move(init_kernels));
            m_kernel_maps.domain_eval[domain].push_back(std::move(eval_kernels));
        }
    }
}

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
template < EquationKernel_c Kernel, ArrayOf_c< size_t > auto field_inds, size_t n_fields, AssemblyOptions asm_opts >
void MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::assembleProblem(
    const Kernel&                        kernel,
    const util::ArrayOwner< d_id_t >&    domain_ids,
    const post::FieldAccess< n_fields >& field_access,
    util::ConstexprValue< field_inds >   field_inds_ctwrpr,
    util::ConstexprValue< asm_opts >     asm_options,
    val_t                                time)
    requires(Kernel::parameters.n_rhs == n_rhs)
{
    pushKernel(kernel, domain_ids, field_access, field_inds_ctwrpr, asm_options, time);
}

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
void MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::describe(std::ostream& out) const
{
    auto local_sizes_min = std::array{m_operator_map->getLocalNumElements(), m_diagonal.size()};
    auto local_sizes_max = local_sizes_min;
    auto sizes_sum       = local_sizes_min;
    m_comm->reduceInPlace(local_sizes_min, 0, MPI_MIN);
    m_comm->reduceInPlace(local_sizes_max, 0, MPI_MAX);
    m_comm->reduceInPlace(sizes_sum, 0, MPI_SUM);
    if (m_comm->getRank() == 0)
    {
        out << std::format("The algebraic system has a total number of {} DOFs\n"
                           "Distribution among {} MPI rank(s):\n"
                           "{:<16}|{:^17}|{:^17}|{:^17}|\n"
                           "{:<16}|{:^17}|{:^17}|{:^17}|\n"
                           "{:<16}|{:^17}|{:^17}|{:^17}|\n\n",
                           sizes_sum[0],
                           m_comm->getSize(),
                           "",
                           "* MIN *",
                           "* MAX *",
                           "* TOTAL *",
                           "OWNED",
                           local_sizes_min[0],
                           local_sizes_max[0],
                           sizes_sum[0],
                           "OWNED + SHARED",
                           local_sizes_min[1],
                           local_sizes_max[1],
                           sizes_sum[1]);
    }
}

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
void MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::beginAssembly()
{
    L3STER_PROFILE_FUNCTION;
    if (m_state == State::OpenForAssembly)
        return;
    m_rhs_view = m_rhs->getLocalViewHost(Tpetra::Access::ReadWrite);
    initKernelMaps();
    m_state = State::OpenForAssembly;
}

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
void MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::endAssembly()
{
    L3STER_PROFILE_FUNCTION;
    assertState(State::OpenForAssembly, "`endAssembly()` was called more than once");
    importDirichletBCs();
    computeDiagAndRhs();
    m_rhs_view = {};
    m_state    = State::Closed;
}

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
void MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::computeDiagAndRhs()
{
    L3STER_PROFILE_FUNCTION;
    const auto visit_border_domain = [&](d_id_t domain) {
        const auto& dom_kernels = m_kernel_maps.domain_init.at(domain);
        const auto  init_border =
            [this, &dom_kernels]< mesh::ElementType ET, el_o_t EO >(const mesh::LocalElementView< ET, EO >& element) {
                for (const auto& [_, border] : dom_kernels)
                    std::invoke(border, element);
                detail::invokeBoundaryKernels(element, &BoundaryInitKernels::border, m_kernel_maps.boundary_init);
            };
        m_border_mesh.visit(init_border, std::views::single(domain), std::execution::par);
    };
    const auto visit_interior_domain = [&](d_id_t domain) {
        const auto& dom_kernels = m_kernel_maps.domain_init.at(domain);
        const auto  init_interior =
            [this, &dom_kernels]< mesh::ElementType ET, el_o_t EO >(const mesh::LocalElementView< ET, EO >& element) {
                for (const auto& [interior, _] : dom_kernels)
                    std::invoke(interior, element);
                detail::invokeBoundaryKernels(element, &BoundaryInitKernels::interior, m_kernel_maps.boundary_init);
            };
        m_interior_mesh.visit(init_interior, std::views::single(domain), std::execution::par);
    };
    const auto handle_dirichlet_dof = [&](local_dof_t dof) {
        m_diagonal[dof] = 1.;
        for (local_dof_t rhs = 0; rhs != n_rhs; ++rhs)
            m_rhs_view(dof, rhs) = m_dirichlet_values(dof, rhs);
    };

    const auto num_owned  = m_node_dof_map.getNumOwnedDofs();
    const auto num_shared = m_node_dof_map.getNumSharedDofs();
    const auto domains    = util::ArrayOwner{m_mesh->getDomainIds()};

    m_rhs->putScalar(0.);
    std::ranges::fill(m_diagonal, 0.);
    zeroExportBuf();

    auto diag_export = comm::Export< val_t >{m_export->getContext(), 1};
    diag_export.setOwned(m_diagonal, m_diagonal.size());
    diag_export.setShared(std::span{m_diagonal}.subspan(num_owned), num_shared);
    m_export->setOwned(m_rhs_view);
    m_export->setShared(m_export_shared_buf);

    m_export->postRecvs(*m_comm);
    diag_export.postRecvs(*m_comm);
    util::tbb::parallelFor(domains, visit_border_domain);
    m_export->postSends(*m_comm);
    diag_export.postSends(*m_comm);
    util::tbb::parallelFor(domains, visit_interior_domain);
    m_export->wait(util::AtomicSumInto{});
    diag_export.wait(util::AtomicSumInto{});
    if (m_dirichlet_bc)
        util::tbb::parallelFor(m_dirichlet_bc->getOwnedDirichletDofs(), handle_dirichlet_dof);
}

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
auto MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::getOperator() const
    -> Teuchos::RCP< const tpetra_operator_t >
{
    assertState(State::Closed, "`getOperator()` was called before `endAssembly()`");
    return util::makeTeuchosRCP< const Operator >(this);
}

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
auto MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::getRhs() const
    -> Teuchos::RCP< const tpetra_multivector_t >
{
    assertState(State::Closed, "`getRhs()` was called before `endAssembly()`");
    return m_rhs;
}

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
auto MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::getSolution() const
    -> Teuchos::RCP< tpetra_multivector_t >
{
    assertState(State::Closed, "`getSolution()` was called before `endAssembly()`");
    return m_solution;
}

namespace detail
{
template < el_o_t... orders, size_t max_dofs_per_node >
auto splitBorderAndInterior(const mesh::MeshPartition< orders... >&              mesh,
                            const dofs::NodeToGlobalDofMap< max_dofs_per_node >& node2dofs)
{
    const auto& ownership   = node2dofs.ownership();
    const auto  is_interior = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
        auto valid_dofs = element.getNodes() | std::views::transform(std::cref(node2dofs)) | std::views::join |
                          std::views::filter([](auto dof) { return dof != invalid_global_dof; });
        return std::ranges::all_of(valid_dofs, [&](auto dof) { return ownership.isOwned(dof); });
    };
    return mesh::splitMeshPartition(mesh, is_interior);
}
} // namespace detail

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::MatrixFreeSystem(
    std::shared_ptr< const MpiComm >                          comm,
    std::shared_ptr< const mesh::MeshPartition< orders... > > mesh,
    const ProblemDefinition< max_dofs_per_node >&             problem_def,
    const BCDefinition< max_dofs_per_node >&                  bc_def)
    : m_comm{std::move(comm)}, m_mesh{std::move(mesh)}, m_state{State::OpenForAssembly}
{
    const auto periodic_bc = bcs::PeriodicBC{bc_def.getPeriodic(), *m_mesh, *m_comm};
    const auto node2dof    = dofs::NodeToGlobalDofMap{*m_comm, *m_mesh, problem_def, periodic_bc, no_condensation_tag};
    const auto [interior, border] = detail::splitBorderAndInterior(*m_mesh, node2dof);
    m_interior_mesh               = mesh::LocalMeshView{interior, *m_mesh};
    m_border_mesh                 = mesh::LocalMeshView{border, *m_mesh};
    const auto& dof_ownership     = node2dof.ownership();
    m_node_dof_map                = dofs::LocalDofMap{node2dof, *m_mesh};
    const auto teuchos_comm       = util::makeTeuchosRCP< Teuchos::MpiComm< int > >(m_comm->get());
    const auto num_all_dofs       = dof_ownership.getOwnershipDist(*m_comm).back();
    m_operator_map                = detail::makeTpetraMapOwned(dof_ownership, teuchos_comm, num_all_dofs);
    m_rhs                         = util::makeTeuchosRCP< tpetra_multivector_t >(m_operator_map, n_rhs);
    m_solution                    = util::makeTeuchosRCP< tpetra_multivector_t >(m_operator_map, n_rhs);
    m_rhs_view                    = m_rhs->getLocalViewHost(Tpetra::Access::ReadWrite);
    m_diagonal                    = util::ArrayOwner< val_t >(dof_ownership.localSize());
    const auto context            = std::make_shared< comm::ImportExportContext >(*m_comm, dof_ownership);
    m_import                      = std::make_unique< comm::Import< val_t > >(context, n_rhs);
    m_export                      = std::make_unique< comm::Export< val_t > >(context, n_rhs);
    m_import_shared_buf           = view_t("import buf", dof_ownership.shared().size(), n_rhs);
    m_export_shared_buf           = view_t("export buf", dof_ownership.shared().size(), n_rhs);
    initKernelMaps();
    const auto& dirichlet = bc_def.getDirichlet();
    if (not dirichlet.empty())
    {
        m_dirichlet_bc.emplace(m_node_dof_map, m_interior_mesh, m_border_mesh, *m_comm, context, dirichlet);
        if (not m_dirichlet_bc->isEmpty()) // Skip allocation if no BC DOFs present in partition
            m_dirichlet_values = view_t("Dirichlet values", dof_ownership.localSize(), n_rhs);
    }
}

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
void MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::applyImpl(const tpetra_multivector_t& x,
                                                                        tpetra_multivector_t&       y,
                                                                        val_t                       alpha,
                                                                        val_t                       beta) const
{
    // *Implementation note*
    // The idea here is to immediately start evaluating the interior, but switch all threads to the border as soon as
    // importing X completes. Once the border is done, the export of Y can be posted and the interior resumed. This is
    // achieved by means of TBB task arenas of different priorities. Note that this approach avoids explicit
    // synchronization - workers are free to resume evaluating the interior as soon as all border work is assigned, they
    // don't need to wait for the border to actually finish.

    L3STER_PROFILE_REGION_BEGIN("Evaluate matrix-free operator");
    util::throwingAssert(x.getNumVectors() == n_rhs);
    util::throwingAssert(y.getNumVectors() == n_rhs);
    beta == 0. ? y.putScalar(0.) : y.scale(beta);
    const auto x_view  = x.getLocalViewHost(Tpetra::Access::ReadOnly);
    const auto y_view  = y.getLocalViewHost(Tpetra::Access::ReadWrite);
    const auto domains = util::ArrayOwner{m_mesh->getDomainIds()};

    zeroExportBuf();
    m_import->setOwned(x_view);
    m_import->setShared(m_import_shared_buf);
    m_export->setOwned(y_view);
    m_export->setShared(m_export_shared_buf);
    m_import->postComms(*m_comm);
    m_export->postRecvs(*m_comm);

    const auto n_cores       = util::GlobalResource< util::hwloc::Topology >::getMaybeUninitialized().getNCores();
    const auto max_par_guard = util::MaxParallelismGuard{n_cores};
    oneapi::tbb::task_arena hp_arena{oneapi::tbb::task_arena::automatic, 1, oneapi::tbb::task_arena::priority::high};
    oneapi::tbb::task_group interior_tasks, border_tasks;

    const auto apply_border = [&] {
        const auto visit_border_domain = [&](d_id_t domain) {
            const auto& dom_kernels = m_kernel_maps.domain_eval.at(domain);
            const auto  eval_border =
                [&]< mesh::ElementType ET, el_o_t EO >(const mesh::LocalElementView< ET, EO >& element) {
                    for (const auto& [_, border] : dom_kernels)
                        std::invoke(border, element, x_view, y_view, alpha);
                    detail::invokeBoundaryKernels(
                        element, &BoundaryEvalKernels::border, m_kernel_maps.boundary_eval, x_view, y_view, alpha);
                };
            m_border_mesh.visit(eval_border, std::views::single(domain), std::execution::par);
        };
        util::tbb::parallelFor(domains, visit_border_domain);
        m_export->postSends(*m_comm);
    };
    const auto apply_interior = [&] {
        const auto visit_interior_domain = [&](d_id_t domain) {
            const auto& dom_kernels = m_kernel_maps.domain_eval.at(domain);
            const auto  eval_interior =
                [&]< mesh::ElementType ET, el_o_t EO >(const mesh::LocalElementView< ET, EO >& element) {
                    for (const auto& [interior, _] : dom_kernels)
                        std::invoke(interior, element, x_view, y_view, alpha);
                    detail::invokeBoundaryKernels(
                        element, &BoundaryEvalKernels::interior, m_kernel_maps.boundary_eval, x_view, y_view, alpha);
                    if (not m_import->testReceive() and m_import->tryReceive())
                        hp_arena.execute([&] { border_tasks.run_and_wait([&] { apply_border(); }); });
                };
            m_interior_mesh.visit(eval_interior, std::views::single(domain), std::execution::par);
        };
        const auto handle_dirichlet_dof = [&](local_dof_t dof) {
            for (local_dof_t rhs = 0; rhs != n_rhs; ++rhs)
            {
                // Export can update Dirichlet DOFs (so we need atomic access), but it will only ever contribute zeros
                // Atomic load -> regular add -> atomic store is faster than CAS loop, the result is preserved
                auto       dest      = std::atomic_ref{y_view(dof, rhs)};
                const auto old_value = dest.load(std::memory_order_relaxed);
                const auto increment = x_view(dof, rhs) * alpha;
                const auto new_value = old_value + increment;
                dest.store(new_value, std::memory_order_relaxed);
            }
        };

        util::tbb::parallelFor(domains, visit_interior_domain);
        L3STER_PROFILE_REGION_BEGIN("Impose Dirichlet BCs");
        if (m_dirichlet_bc)
            util::tbb::parallelFor(m_dirichlet_bc->getOwnedDirichletDofs(), handle_dirichlet_dof);
        L3STER_PROFILE_REGION_END("Impose Dirichlet BCs");
    };
    interior_tasks.run_and_wait([&] { apply_interior(); });
    hp_arena.execute([&] { border_tasks.wait(); });
    if (not m_import->testReceive())
    {
        m_import->waitReceive();
        apply_border();
    }

    m_export->wait(util::AtomicSumInto{});
    m_import->wait();
    L3STER_PROFILE_REGION_END("Evaluate matrix-free operator");
}

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
void MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::copyDiagImpl(tpetra_vector_t& diag) const
{
    const auto num_owned_dofs = m_node_dof_map.getNumOwnedDofs();
    util::throwingAssert(diag.getLocalLength() == num_owned_dofs);
    auto diag_view = diag.getLocalViewHost(Tpetra::Access::OverwriteAll);
    std::ranges::copy(m_diagonal | std::views::take(num_owned_dofs), diag_view.data());
}

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
void MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::assertState(State                expected,
                                                                          std::string_view     err_msg,
                                                                          std::source_location src_loc) const
{
    util::throwingAssert(m_state == expected, err_msg, src_loc);
}

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
template < ResidualKernel_c Kernel, std::integral dofind_t, size_t n_fields >
void MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::setValues(
    const Teuchos::RCP< tpetra_multivector_t >&                   vector,
    const Kernel&                                                 kernel,
    const util::ArrayOwner< d_id_t >&                             domain_ids,
    const std::array< dofind_t, Kernel::parameters.n_equations >& dof_inds,
    const post::FieldAccess< n_fields >&                          field_access,
    val_t                                                         time)
{
    util::throwingAssert(vector->getNumVectors() == n_rhs);
    computeValuesAtNodes(kernel,
                         *m_comm,
                         m_interior_mesh,
                         m_border_mesh,
                         *m_export,
                         domain_ids,
                         m_node_dof_map,
                         dof_inds,
                         field_access,
                         vector->getLocalViewHost(Tpetra::Access::ReadWrite),
                         time);
}

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
template < ResidualKernel_c Kernel, std::integral dofind_t, size_t n_fields >
void MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::setDirichletBCValues(
    const Kernel&                                                 kernel,
    const util::ArrayOwner< d_id_t >&                             domain_ids,
    const std::array< dofind_t, Kernel::parameters.n_equations >& dof_inds,
    const post::FieldAccess< n_fields >&                          field_access,
    val_t                                                         time)
{
    util::throwingAssert(m_dirichlet_bc.has_value(), "`setDirichletBCValues` called but no Dirichlet BCs were defined");
    if (not m_dirichlet_bc->isEmpty())
    {
        const auto owned_range = std::pair< size_t, size_t >{0, m_operator_map->getLocalNumElements()};
        computeValuesAtNodes(kernel,
                             *m_comm,
                             m_interior_mesh,
                             m_border_mesh,
                             *m_export,
                             domain_ids,
                             m_node_dof_map,
                             dof_inds,
                             field_access,
                             Kokkos::subview(m_dirichlet_values, owned_range, Kokkos::ALL),
                             time);
        m_bcs_need_import = true;
    }
}

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
template < size_t n_vals, std::integral dofind_t >
void MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::setDirichletBCValues(
    const std::array< val_t, n_vals >&    values,
    const util::ArrayOwner< d_id_t >&     domain_ids,
    const std::array< dofind_t, n_vals >& dof_inds)
    requires(n_rhs == 1)
{
    util::throwingAssert(m_dirichlet_bc.has_value(), "`setDirichletBCValues` called but no Dirichlet BCs were defined");
    if (not m_dirichlet_bc->isEmpty())
    {
        const auto vals_to_set = std::array{std::span{values}};
        const auto owned_range = std::pair< size_t, size_t >{0, m_operator_map->getLocalNumElements()};
        computeValuesAtNodes(*m_comm,
                             m_interior_mesh,
                             m_border_mesh,
                             *m_export,
                             domain_ids,
                             m_node_dof_map,
                             dof_inds,
                             vals_to_set,
                             Kokkos::subview(m_dirichlet_values, owned_range, Kokkos::ALL));
        m_bcs_need_import = true;
    }
}

template < size_t max_dofs_per_node, size_t n_rhs, el_o_t... orders >
void MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >::updateSolution(
    const util::ArrayOwner< size_t >& sol_inds,
    SolutionManager&                  sol_man,
    const util::ArrayOwner< size_t >& sol_man_inds)
{
    util::throwingAssert(sol_man_inds.size() == sol_inds.size() * n_rhs,
                         "Source and destination indices lengths must match");
    util::throwingAssert(std::ranges::none_of(sol_inds, [](size_t i) { return i >= max_dofs_per_node; }),
                         "Source index out of bounds");
    util::throwingAssert(std::ranges::none_of(sol_man_inds, [&](size_t i) { return i >= sol_man.nFields(); }),
                         "Destination index out of bounds");

    const auto solution_view   = m_solution->getLocalViewHost(Tpetra::Access::ReadOnly);
    const auto solution_access = BorderAccessor{solution_view, const_view_t{m_import_shared_buf}};
    const auto sol_man_view    = sol_man.getRawView();
    const auto save_elem_vals  = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::LocalElementView< ET, EO >& elem) {
        for (n_loc_id_t node : elem.getLocalNodes())
        {
            const auto& node_dofs = m_node_dof_map(node);
            for (auto&& [i, sol_ind] : sol_inds | std::views::enumerate)
            {
                const auto dof = node_dofs[sol_ind];
                if (dofs::LocalDofMap< max_dofs_per_node >::isValid(dof))
                    for (local_dof_t rhs = 0; rhs != n_rhs; ++rhs)
                    {
                        const auto value      = solution_access(dof, rhs);
                        const auto dest_field = sol_man_inds[i * n_rhs + rhs];
                        auto&      dest       = sol_man_view(node, dest_field);
                        std::atomic_ref{dest}.store(value, std::memory_order_relaxed);
                    }
            }
        }
    };

    m_import->setOwned(solution_view);
    m_import->setShared(m_import_shared_buf);
    m_import->postComms(*m_comm);
    m_interior_mesh.visit(save_elem_vals, std::execution::par);
    m_import->waitReceive();
    m_border_mesh.visit(save_elem_vals, std::execution::par);
    m_import->wait();
}
} // namespace lstr::algsys
#endif // L3STER_ALGSYS_MATRIXFREESYSTEM_HPP
