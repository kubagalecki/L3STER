// Implementation of the abstract factory design pattern

#pragma once

#include <map>
#include <exception>
#include <utility>
#include <memory>

namespace lstr
{
	namespace despat
	{
		template<typename KeyType, typename CreatorBase>
		class Factory final
		{
			// ALIASES
			using map_t			= std::map<KeyType, std::unique_ptr< CreatorBase >>;
			using ptr_t			= typename CreatorBase::ptr_t;

		public:
			// METHODS
			// Factory is a static class and cannot be instantiated
			Factory()											= delete;
			Factory(const Factory&)								= delete;
			Factory& operator=(const Factory&)					= delete;
			Factory(const Factory&&)							= delete;
			Factory& operator=(const Factory&&)					= delete;
			~Factory()											= delete;

			static void								registerCreator(const KeyType&, ptr_t);
			static void								unregisterCreator(const KeyType&);
			static ptr_t		getCreator(const KeyType&);

		private:
			// MEMBERS
			static map_t		creator_map;
		};

		template<typename KeyType, typename CreatorBase>
		typename Factory<KeyType, CreatorBase>::map_t
			Factory<KeyType, CreatorBase>::
			creator_map	= Factory<KeyType, CreatorBase>::map_t{};

		template<typename KeyType, typename CreatorBase>
		void Factory<KeyType, CreatorBase>::
			registerCreator(const KeyType& key, ptr_t creator)
		{
			if (creator_map.find(key) != creator_map.end())
				throw (std::invalid_argument("The creator you are trying to register is already registered\n"));

			creator_map.emplace(typename map_t::value_type{ key, creator });
		}

		template<typename KeyType, typename CreatorBase>
		void Factory<KeyType, CreatorBase>::
			unregisterCreator(const KeyType& key)
		{
			auto it = creator_map.find(key);

			if (it == creator_map.end())
				throw (std::invalid_argument("The creator you are trying to unregister was never registered.\n"));

			if (creator_map.erase(key))
				throw (std::runtime_error("Could not register creator.\n"))
		}

		template<typename KeyType, typename CreatorBase>
		Factory<KeyType, CreatorBase>::ptr_t Factory<KeyType, CreatorBase>::
			getCreator(const KeyType& key)
		{
			auto it = creator_map.find(key);

			if (it == creator_map.end())
				throw (std::invalid_argument("The creator you are trying to access was never registered.\n"));

			return *it;
		}

	}
}