// Data type representing a physical element

#ifndef L3STER_INCGUARD_MESH_ELEMENT_HPP
#define L3STER_INCGUARD_MESH_ELEMENT_HPP

#include "mesh/ElementTraits.hpp"
#include "mesh/ReferenceElement.hpp"
#include "typedefs/Types.h"
#include "utility/Meta.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lstr::mesh {
    namespace helpers {
// HELPER FOR GENERATING {1, 2, 3, ..., N_NODES} ARRAY
        template<size_t SIZE>
        class NodeOrderHelper {
            using array_t = std::array<size_t, SIZE>;

        public:
            static const array_t &get();

        private:
            static array_t generate();

            static const array_t unsorted_order;
        };

        template<size_t SIZE>
        const typename NodeOrderHelper<SIZE>::array_t &NodeOrderHelper<SIZE>::get() {
            return unsorted_order;
        }

        template<size_t SIZE>
        typename NodeOrderHelper<SIZE>::array_t NodeOrderHelper<SIZE>::generate() {
            auto &&ret_val = array_t{{0}};
            std::iota(ret_val.begin(), ret_val.end(), 0);
            return ret_val;
        }

        template<size_t SIZE>
        const typename NodeOrderHelper<SIZE>::array_t NodeOrderHelper<SIZE>::unsorted_order{
                NodeOrderHelper<SIZE>::generate()};
    } // namespace helpers

    class ElementBase {
    };

    template<ElementTypes ELTYPE, types::el_o_t ELORDER>
    class Element final : public ElementBase {
    private:
        // private static ("alias for variable")
        static constexpr const size_t n_nodes{ReferenceElement<ELTYPE, ELORDER>::getNumberOfNodes()};

    public:
        // ALIASES
        using node_array_t = std::array<types::n_id_t, n_nodes>;
        using node_array_ref_t = node_array_t &;
        using node_array_constref_t = const node_array_t &;

        // CONSTRUCTORS
        Element() = default;

        Element(node_array_constref_t _nodes);

        // GETTERS
        node_array_constref_t getNodes() const;

    private:
        // SORT
        void sort();

        // MEMBERS
        // nodes
        const node_array_t nodes = node_array_t{};
        std::array<size_t, n_nodes> node_order;

        // element data
        typename ElementTraits<Element<ELTYPE, ELORDER>>::ElementData data;
    };

    template<ElementTypes ELTYPE, types::el_o_t ELORDER>
    Element<ELTYPE, ELORDER>::Element(node_array_constref_t _nodes)
            : nodes(_nodes), node_order(helpers::NodeOrderHelper<n_nodes>::get()) {
        this->sort();
    }

    template<ElementTypes ELTYPE, types::el_o_t ELORDER>
    typename Element<ELTYPE, ELORDER>::node_array_constref_t Element<ELTYPE, ELORDER>::getNodes() const {
        return nodes;
    }

    template<ElementTypes ELTYPE, types::el_o_t ELORDER>
    void Element<ELTYPE, ELORDER>::sort() {
        std::sort(node_order.begin(), node_order.end(), [this](size_t n1, size_t n2) {
            return this->nodes[n1] < this->nodes[n2];
        });
    }

// Define alias for easy templating over all possible element/order combinations
    template<template<typename...> typename T>
    using TemplateOverAllElements = typename util::meta::
    cartesian_product_t<T, Element, ElementTypesArray, ElementOrdersArray>::type;

} // namespace lstr::mesh

#endif // end include guard
