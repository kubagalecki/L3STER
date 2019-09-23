#pragma once

#include "ReferenceElement.hpp"
#include "Types.h"

#include <array>
#include <vector>
#include <stdexcept>
#include <string>
#include <typeinfo>
//#include <iostream>
#include <algorithm>
#include <numeric>
#include <utility>

namespace lstr
{
	namespace mesh
	{
		namespace helpers
		{
			// HELPER FOR GENERATING {1, 2, 3, ..., N_NODES} ARRAY
			template <size_t SIZE>
			class NodeOrderHelper
			{
				using array_t = std::array<size_t, SIZE>;

			public:
				static const array_t& get() { return unsorted_order; }

			private:
				static array_t			generate();
				static const array_t	unsorted_order;
			};

			template <size_t SIZE>
			typename NodeOrderHelper<SIZE>::array_t NodeOrderHelper<SIZE>::generate()
			{
				array_t ret_val{ {0} };
				std::iota(ret_val.begin(), ret_val.end(), 0);
				return ret_val;
			}

			template <size_t SIZE>
			const typename NodeOrderHelper<SIZE>::array_t unsorted_order = NodeOrderHelper<SIZE>::generate();
		}

		template <ElementTypes ELTYPE, types::el_o_t ELORDER>
		class Element final
		{
		public:

			// ALIASES
			static constexpr const size_t n_nodes	= ReferenceElement<ELTYPE, ELORDER>::getNumberOfNodes();
			using node_array						= std::array <types::n_id_t, n_nodes>;
			using node_array_ref					= node_array &;
			using node_array_constref				= const node_array &;

			// CONSTRUCTORS
			Element()								= delete;
			Element(node_array _nodes);

			// GETTERS
			node_array_constref	getNodes() const;

			// SETTERS
			void				setNode(size_t, types::n_id_t);
			
			// SORT
			void				sort();

		private:

			// MEMBERS
			const node_array				nodes;
			std::array<size_t, n_nodes>		node_order;
		};

		//template <ElementTypes ELTYPE, types::el_o_t ELORDER>
		//Element<ELTYPE, ELORDER>::Element()						: nodes{ { 0 } },
		//	node_order(helpers::NodeOrderHelper<n_nodes>::get())
		//{
		//	
		//}

		template <ElementTypes ELTYPE, types::el_o_t ELORDER>
		Element<ELTYPE, ELORDER>::Element(node_array _nodes)	: nodes(_nodes),
			node_order(helpers::NodeOrderHelper<n_nodes>::get())
		{
			this->sort();
		}

		template <ElementTypes ELTYPE, types::el_o_t ELORDER>
		typename Element<ELTYPE, ELORDER>::node_array_constref Element<ELTYPE, ELORDER>::getNodes() const
		{
			return nodes;
		}

		template <ElementTypes ELTYPE, types::el_o_t ELORDER>
		void Element<ELTYPE, ELORDER>::setNode(size_t n_no, types::n_id_t n_id)
		{
			if (n_no >= nodes.size())
			{
				std::string exceptTxt = static_cast<std::string>("Requested set node ") + std::to_string(n_no)
					+ static_cast<std::string>(". Element ") + typeid(Element<ELTYPE, ELORDER>).name()
					+ static_cast<std::string>(" has only ") + std::to_string(nodes.size())
					+ static_cast<std::string>(" nodes.");
				throw (std::out_of_range(exceptTxt));
				return;
			}

			nodes[n_no] = n_id;
		}

		template <ElementTypes ELTYPE, types::el_o_t ELORDER>
		void Element<ELTYPE, ELORDER>::sort()
		{
			std::sort(node_order.begin(), node_order.end(),
				[this] (size_t n1, size_t n2) { return this->nodes[n1] < this->nodes[n2]; });

			///////////////////////////////////////////////////////////////////////////////////////////////
			// OLD CODE FOR ACTUALLY SORTING THE NODES, AS OPPOSED TO JUST GENERATING THE REORDER VECTOR //
			///////////////////////////////////////////////////////////////////////////////////////////////

			//// Aliases
			//using aux_p_t = std::pair<types::n_id_t, size_t>;
			//using aux_a_t = std::array<aux_p_t, n_nodes>;

			//// Group nodes IDs and their indices into a vector of pairs
			//aux_a_t aux_a;
			//std::transform(nodes.begin(), nodes.end(), node_order.begin(),
			//	aux_a.begin(), [](types::n_id_t n, size_t o) -> aux_p_t { return std::make_pair(n, o); });

			//// Sort
			//std::sort(aux_a.begin(), aux_a.end());

			//// Move data back to Element class members
			//std::transform(aux_a.begin(), aux_a.end(), nodes.begin(),
			//	[](aux_p_t p) -> types::n_id_t { return p.first; });
			//std::transform(aux_a.begin(), aux_a.end(), node_order.begin(),
			//	[](aux_p_t p) -> size_t { return p.second; });
		}
	}
}