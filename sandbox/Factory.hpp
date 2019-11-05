// Implementation of the abstract factory design pattern

#pragma once

#include <map>
#include <stdexcept>
#include <utility>
#include <memory>

namespace lstr
{
    namespace despat
    {
        //////////////////////////////////////////////////////////////////////////////////////////////
        //									CREATOR BASE CLASS										//
        //////////////////////////////////////////////////////////////////////////////////////////////
        template <typename BaseT>
        class CreatorBase
        {
        public:
            using						s_ptr_t					= std::shared_ptr<BaseT>;
            using						u_ptr_t					= std::unique_ptr<BaseT>;
            virtual u_ptr_t				create() const			= 0;
            virtual						~CreatorBase()			= default;

        protected:
            CreatorBase()						= default;
            CreatorBase(const CreatorBase&)		= default;
            CreatorBase(CreatorBase&&)			= default;
        };

        //////////////////////////////////////////////////////////////////////////////////////////////
        //										CREATOR CLASS										//
        //////////////////////////////////////////////////////////////////////////////////////////////
        template<typename KeyType, typename ProductBase, typename Product>
        class Creator final : public CreatorBase<ProductBase>
        {
        public:
            using s_ptr_t = std::shared_ptr<const Creator>;

            // CTORS & DTORS
							Creator()					= default;
            virtual			~Creator() override			= default;

            // METHODS
			static const s_ptr_t&					getInstance()				{ return instance; }
            std::unique_ptr<ProductBase>			create() const final		{ return std::make_unique<Product>(); }

        private:
            // Creator is a singleton - even if it's registered with multiple keys, it still creates objects
            // of a given type
            static s_ptr_t instance;
        };

        template<typename KeyType, typename ProductBase, typename Product>
        std::shared_ptr<const Creator< KeyType, ProductBase, Product >> Creator<KeyType, ProductBase, Product>::
        instance = std::make_shared<const Creator< KeyType, ProductBase, Product >>();

        //////////////////////////////////////////////////////////////////////////////////////////////
        //										FACTORY CLASS										//
        //////////////////////////////////////////////////////////////////////////////////////////////
        template<typename KeyType, typename ProductBase>
        class Factory final
        {
            // ALIASES
            using ptr_t			= std::shared_ptr<const CreatorBase< ProductBase >>;
            using map_t			= std::map<KeyType, ptr_t>;

        public:
            // METHODS
            // Factory is a static class and cannot be instantiated
            Factory()											= delete;
            Factory(const Factory&)								= delete;
            Factory& operator=(const Factory&)					= delete;
            Factory(const Factory&&)							= delete;
            Factory& operator=(const Factory&&)					= delete;
            ~Factory()											= delete;

            template<typename Product>
            static void								registerCreator(const KeyType&);
            static void								unregisterCreator(const KeyType&);
            static const ptr_t&						getCreator(const KeyType&);
            static std::unique_ptr<ProductBase>		create(const KeyType& key)			{ return getCreator(key)->create(); }

        private:
            // MEMBERS
            static map_t		creator_map;
        };

        // initialize creator_map as empty
        template<typename KeyType, typename ProductBase>
        typename Factory<KeyType, ProductBase>::map_t
        Factory<KeyType, ProductBase>::
        creator_map	= Factory<KeyType, ProductBase>::map_t{};

        template<typename KeyType, typename ProductBase>
        template<typename Product>
        void Factory<KeyType, ProductBase>::registerCreator(const KeyType& key)
        {
            static_assert(std::is_base_of<ProductBase, Product>::value,
				"The product type you are trying to register must be derived from the declared base class\n");

            if (creator_map.find(key) != creator_map.end())
                throw (std::invalid_argument("The creator you are trying to register is already registered\n"));

            creator_map[key] = Creator<KeyType, ProductBase, Product>::getInstance();
        }

        template<typename KeyType, typename CreatorBase>
        void Factory<KeyType, CreatorBase>::
        unregisterCreator(const KeyType& key)
        {
            auto it = creator_map.find(key);

            if (it == creator_map.end())
                throw (std::invalid_argument("The creator you are trying to unregister was never registered.\n"));

			creator_map.erase(key);
        }

        template<typename KeyType, typename CreatorBase>
        const typename Factory<KeyType, CreatorBase>::ptr_t& Factory<KeyType, CreatorBase>::
        getCreator(const KeyType& key)
        {
            auto it = creator_map.find(key);

            if (it == creator_map.end())
                throw (std::invalid_argument("The creator you are trying to access was never registered.\n"));

            return it->second;
        }
    }
}
