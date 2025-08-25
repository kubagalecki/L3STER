#ifndef L3STER_BCS_BCDEFINITION_HPP
#define L3STER_BCS_BCDEFINITION_HPP

#include "l3ster/common/ProblemDefinition.hpp"
#include "l3ster/util/Algorithm.hpp"

namespace lstr
{
namespace bcs
{
template < size_t max_dofs_per_node >
class DirichletBCDefinition
{
    struct Def
    {
        util::ArrayOwner< d_id_t > domains;
        util::ArrayOwner< size_t > dof_inds;
    };

public:
    void defineDirichletBoundary(util::ArrayOwner< d_id_t > domains, util::ArrayOwner< size_t > dof_inds)
    {
        util::throwingAssert(std::ranges::all_of(dof_inds, [](auto i) { return i < max_dofs_per_node; }));
        m_defs.push_back(Def{util::getUniqueCopy(std::move(domains)), util::getUniqueCopy(std::move(dof_inds))});
    }
    void normalize(size_t dof_ind) { m_to_normalize.set(dof_ind); }

    auto begin() const { return m_defs.begin(); }
    auto end() const { return m_defs.end(); }
    auto size() const { return m_defs.size(); }
    bool empty() const { return m_defs.empty(); }

    auto getNormalized() const { return util::getTrueInds(m_to_normalize); }

private:
    std::vector< Def >               m_defs;
    std::bitset< max_dofs_per_node > m_to_normalize;
};

template < size_t max_dofs_per_node >
class PeriodicBCDefinition
{
    struct Def
    {
        util::ArrayOwner< d_id_t >       boundary_ids_src, boundary_ids_dest;
        std::array< val_t, 3 >           translation;
        std::bitset< max_dofs_per_node > dofs;
    };

public:
    val_t tolerance = 1e-12;

    void definePeriodicBoundary(util::ArrayOwner< d_id_t >        boundaries_src,
                                util::ArrayOwner< d_id_t >        boundaries_dest,
                                std::array< val_t, 3 >            translation,
                                const util::ArrayOwner< size_t >& dof_inds)
    {
        util::throwingAssert(std::ranges::all_of(dof_inds, [](auto i) { return i < max_dofs_per_node; }));
        auto dof_bitset = std::bitset< max_dofs_per_node >{};
        for (auto d : dof_inds)
            dof_bitset.set(d);
        m_defs.push_back(Def{util::getUniqueCopy(std::move(boundaries_src)),
                             util::getUniqueCopy(std::move(boundaries_dest)),
                             translation,
                             dof_bitset});
    }

    auto begin() const { return m_defs.begin(); }
    auto end() const { return m_defs.end(); }
    auto size() const { return m_defs.size(); }
    bool empty() const { return m_defs.empty(); }

private:
    std::vector< Def > m_defs;
};
} // namespace bcs

template < size_t max_dofs_per_node >
class BCDefinition
{
public:
    BCDefinition() = default;
    explicit BCDefinition(const ProblemDefinition< max_dofs_per_node >&) {}

    /// Dirichlet boundary at specified IDs, affects the specified DOFs (all by default)
    void defineDirichlet(util::ArrayOwner< d_id_t > boundaries,
                         util::ArrayOwner< size_t > dof_inds = std::views::iota(0uz, max_dofs_per_node))
    {
        m_dirichlet.defineDirichletBoundary(std::move(boundaries), std::move(dof_inds));
    }

    /// Periodic boundary condition between specified boundaries, affects the specified DOFs (all by default)
    /// Each node at X belonging to source must have a corresponding node at X + translation belonging to destination
    /// Source and destination can be freely swapped (as long as the translation is negated) - this choice is arbitrary
    void definePeriodic(util::ArrayOwner< d_id_t > bounds_src,
                        util::ArrayOwner< d_id_t > bounds_dest,
                        std::array< val_t, 3 >     trans,
                        util::ArrayOwner< size_t > dof_inds = std::views::iota(0uz, max_dofs_per_node))
    {
        m_periodic.definePeriodicBoundary(std::move(bounds_src), std::move(bounds_dest), trans, std::move(dof_inds));
    }

    /// Condition for matching nodes: |src + translation - dest| < tolerance. Default value is 1e-12.
    void setPeriodicMatchTolerance(val_t tolerance) { m_periodic.tolerance = tolerance; }

    /// Normalize the value of the specified unknowns in the domain, equivalent to homogeneous Dirichlet BC for one
    /// arbitrarily chosen node
    void normalize(const util::ArrayOwner< size_t >& dof_inds)
    {
        constexpr auto less_than_max = std::bind_back(std::less{}, max_dofs_per_node);
        util::throwingAssert(std::ranges::all_of(dof_inds, less_than_max));
        for (auto i : dof_inds)
            m_dirichlet.normalize(i);
    }

    auto getDirichlet() const -> const auto& { return m_dirichlet; }
    auto getPeriodic() const -> const auto& { return m_periodic; }

private:
    bcs::DirichletBCDefinition< max_dofs_per_node > m_dirichlet;
    bcs::PeriodicBCDefinition< max_dofs_per_node >  m_periodic;
};
template < size_t dofs_per_node >
BCDefinition(const ProblemDefinition< dofs_per_node >& problem_def) -> BCDefinition< dofs_per_node >;
} // namespace lstr
#endif // L3STER_BCS_BCDEFINITION_HPP
