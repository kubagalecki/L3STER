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
    void defineDirichletBoundary(const util::ArrayOwner< d_id_t >& domains, const util::ArrayOwner< size_t >& dof_inds)
    {
        util::throwingAssert(std::ranges::all_of(dof_inds, [](auto i) { return i < max_dofs_per_node; }));
        m_defs.push_back(Def{util::getUniqueCopy(copy(domains)), util::getUniqueCopy(copy(dof_inds))});
    }

    auto begin() const { return m_defs.begin(); }
    auto end() const { return m_defs.end(); }
    auto size() const { return m_defs.size(); }
    bool empty() const { return m_defs.empty(); }

private:
    std::vector< Def > m_defs;
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
    PeriodicBCDefinition() = default;
    template < ProblemDef problem_def >
    explicit PeriodicBCDefinition(util::ConstexprValue< problem_def >)
        requires(problem_def.n_fields == max_dofs_per_node)
    {}

    val_t tolerance = 1.e-12;

    void definePeriodicBoundary(const util::ArrayOwner< d_id_t >& boundaries_src,
                                const util::ArrayOwner< d_id_t >& boundaries_dest,
                                std::array< val_t, 3 >            translation,
                                const util::ArrayOwner< size_t >& dof_inds)
    {
        util::throwingAssert(std::ranges::all_of(dof_inds, [](auto i) { return i < max_dofs_per_node; }));
        auto dof_bitset = std::bitset< max_dofs_per_node >{};
        for (auto d : dof_inds)
            dof_bitset.set(d);
        m_defs.push_back(Def{util::getUniqueCopy(copy(boundaries_src)),
                             util::getUniqueCopy(copy(boundaries_dest)),
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
    template < ProblemDef problem_def >
    explicit BCDefinition(util::ConstexprValue< problem_def >)
        requires(problem_def.n_fields == max_dofs_per_node)
    {}

    /// Dirichlet boundary at specified IDs, affects the specified DOFs (all by default)
    void defineDirichlet(const util::ArrayOwner< d_id_t >& boundaries,
                         const util::ArrayOwner< size_t >& dof_inds = std::views::iota(0uz, max_dofs_per_node))
    {
        m_dirichlet.defineDirichletBoundary(boundaries, dof_inds);
    }

    /// Periodic boundary condition between specified boundaries, affects the specified DOFs (all by default)
    /// Each node at X belonging to source must have a corresponding node at X + translation belonging to destination
    /// Source and destination can be freely swapped (as long as the translation is negated) - this choice is arbitrary
    void definePeriodic(const util::ArrayOwner< d_id_t >& boundaries_source,
                        const util::ArrayOwner< d_id_t >& boundaries_destination,
                        std::array< val_t, 3 >            translation,
                        const util::ArrayOwner< size_t >& dof_inds = std::views::iota(0uz, max_dofs_per_node))
    {
        m_periodic.definePeriodicBoundary(boundaries_source, boundaries_destination, translation, dof_inds);
    }

    /// Condition for matching nodes: |src + translation - dest| < tolerance. Default value is 1e-12.
    void setPeriodicMatchTolerance(val_t tolerance) { m_periodic.tolerance = tolerance; }

    auto getDirichlet() const -> const auto& { return m_dirichlet; }
    auto getPeriodic() const -> const auto& { return m_periodic; }

private:
    bcs::DirichletBCDefinition< max_dofs_per_node > m_dirichlet;
    bcs::PeriodicBCDefinition< max_dofs_per_node >  m_periodic;
};
template < ProblemDef problem_def >
BCDefinition(util::ConstexprValue< problem_def >) -> BCDefinition< problem_def.n_fields >;
} // namespace lstr
#endif // L3STER_BCS_BCDEFINITION_HPP
