// Implementation of the abstract factory design pattern

#pragma once

#include <map>
#include <exception>
#include <utility>

namespace lstr
{
	namespace despat
	{
		template<typename ProductBase, typename CreatorBase, typename KeyType>
		class Factory
		{
			// ALIASES
			using map_t = std::map<KeyType, CreatorBase*>;

		public:
			// METHODS

			// Register creator
			void registerCreator(const KeyType& key, CreatorBase* creator)
			{
				if (creator_map.find(key) != creator_map.end())
					throw (std::invalid_argument("The creator you are trying to register is already registered\n"));

				creator_map.emplace(std::make_pair(key, creator)
			}

			// Unregister creator
			void unregisterCreator(const KeyType& key)
			{
				auto it = creator_map.find(key);

				if (it == creator_map.end())
					throw (std::invalid_argument("The creator you are trying to unregister was never registered.\n"));

				if (creator_map.erase(key))
					throw (std::runtime_error("Could not register creator.\n"))
			}

			// Get creator
			CreatorBase* getCreator(const KeyType& key)
			{
				auto it = creator_map.find(key);

				if (it == creator_map.end())
					throw (std::invalid_argument("The creator you are trying to access was never registered.\n"));

				return *it;

			}
		private:
			// MEMBERS
			map_t creator_map;
		};
	}
}