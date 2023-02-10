#ifndef L3STER_DENSEGRAPH_HPP
#define L3STER_DENSEGRAPH_HPP

#include "l3ster/assembly/SparsityGraph.hpp"
#include "l3ster/util/DynamicBitset.hpp"

namespace lstr
{
class DenseGraph
{
public:
    template < auto problem_def >
    DenseGraph(const MeshPartition&                                                        mesh,
               ConstexprValue< problem_def >                                               problemdef_ctwrpr,
               const detail::node_interval_vector_t< detail::deduceNFields(problem_def) >& dof_intervals,
               size_t                                                                      n_dofs)
        : dim{n_dofs}, entries{dim * dim}
    {
        const auto node_to_dof_map = NodeToGlobalDofMap{mesh, dof_intervals};
        const auto process_domain  = [&]< auto dom_def >(ConstexprValue< dom_def >) {
            constexpr auto  domain_id        = dom_def.first;
            constexpr auto& coverage         = dom_def.second;
            constexpr auto  covered_dof_inds = getTrueInds< coverage >();

            const auto process_element = [&]< ElementTypes T, el_o_t O >(const Element< T, O >& element) {
                const auto element_dofs = detail::getSortedElementDofs< covered_dof_inds >(element, node_to_dof_map);
                for (auto row : element_dofs)
                    for (auto col : element_dofs)
                        getRow(row).set(col);
            };
            mesh.visit(process_element, domain_id);
        };
        forConstexpr(process_domain, problemdef_ctwrpr);
    }

    [[nodiscard]] auto getRow(size_t row) { return entries.getSubView(row * dim, (row + 1) * dim); }
    [[nodiscard]] auto getRow(size_t row) const { return entries.getSubView(row * dim, (row + 1) * dim); }

    void print() const
    {
        for (size_t row = 0; row < dim; ++row)
        {
            const auto row_data = getRow(row);
            std::cout << row << ": ";
            for (size_t col = 0; col < dim; ++col)
                if (row_data.test(col))
                    std::cout << col << ' ';
            std::cout << '\n';
        }
    }

private:
    size_t        dim; // assume square
    DynamicBitset entries;
};
} // namespace lstr
#endif // L3STER_DENSEGRAPH_HPP