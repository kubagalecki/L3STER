// Implementation of the abstract factory design pattern

#ifndef L3STER_INCGUARD_UTIL_FACTORY_HPP
#define L3STER_INCGUARD_UTIL_FACTORY_HPP

#include <map>
#include <stdexcept>
#include <utility>
#include <memory>

namespace lstr
{
namespace util
{
//////////////////////////////////////////////////////////////////////////////////////////////
//                                  CREATOR BASE CLASS                                      //
//////////////////////////////////////////////////////////////////////////////////////////////
template <typename ProductBase>
class CreatorBase
{
public:
    using                       s_ptr_t                 = std::shared_ptr<ProductBase>;
    using                       u_ptr_t                 = std::unique_ptr<ProductBase>;
    virtual u_ptr_t             create() const          = 0;
    virtual                     ~CreatorBase()          = default;

protected:
    CreatorBase()                                       = default;
    CreatorBase(const CreatorBase&)                     = default;
    CreatorBase(CreatorBase&&) noexcept                 = default;
    CreatorBase& operator=(const CreatorBase&)          = default;
    CreatorBase& operator=(CreatorBase&&) noexcept      = default;
};

//////////////////////////////////////////////////////////////////////////////////////////////
//                                      CREATOR CLASS                                       //
//////////////////////////////////////////////////////////////////////////////////////////////
template<typename KeyType, typename ProductBase, typename Product>
class Creator final : public CreatorBase<ProductBase>
{
public:
    // CTORS & DTORS
    Creator()                               = default;
    Creator(const Creator&)                 = default;
    Creator(Creator&&) noexcept             = default;
    Creator& operator=(const Creator&)      = default;
    Creator& operator=(Creator&&) noexcept  = default;
    virtual ~Creator() final                = default;

    // METHODS
    std::unique_ptr<ProductBase>    create() const final
    {
        return std::make_unique<Product>();
    }

private:
};

//////////////////////////////////////////////////////////////////////////////////////////////
//                                      FACTORY CLASS                                       //
//////////////////////////////////////////////////////////////////////////////////////////////
template<typename KeyType, typename ProductBase>
class Factory final
{
    // ALIASES
    using creat_t       = CreatorBase<ProductBase>;
    using s_ptr_t       = std::shared_ptr<creat_t>;
    using map_t         = std::map<KeyType, s_ptr_t>;
    using cr_key_t      = const KeyType&;

public:
    // METHODS
    // Factory is a static class and cannot be instantiated
    Factory()                                           = delete;
    Factory(const Factory&)                             = delete;
    Factory& operator=(const Factory&)                  = delete;
    Factory(const Factory&&)                            = delete;
    Factory& operator=(const Factory&&)                 = delete;
    ~Factory()                                          = delete;

    template<typename Product>
    static void                             registerCreator(cr_key_t);
    static void                             unregisterCreator(cr_key_t);
    static s_ptr_t                          getCreator(cr_key_t);
    static std::unique_ptr<ProductBase>     create(cr_key_t key);

private:
    // MEMBERS
    // Creator map lives in getter function - construct on first use
    static map_t&       getCreatorMap();
};

template<typename KeyType, typename ProductBase>
typename Factory<KeyType, ProductBase>::map_t& Factory<KeyType, ProductBase>::getCreatorMap()
{
    static auto creator_map = new map_t{};
    return *creator_map;
}

template<typename KeyType, typename ProductBase>
template<typename Product>
void Factory<KeyType, ProductBase>::registerCreator(const KeyType& key)
{
    static_assert(std::is_base_of<ProductBase, Product>::value,
                  "The product type you are trying to register must be derived from the declared base class\n");

    auto it = getCreatorMap().find(key);

    if (it != getCreatorMap().end())
        throw (std::invalid_argument("The creator you are trying to register is already registered\n"));

    getCreatorMap().insert(std::make_pair(key, std::make_shared< Creator<KeyType, ProductBase, Product> >()));
}

template<typename KeyType, typename ProductBase>
void Factory<KeyType, ProductBase>::
unregisterCreator(const KeyType& key)
{
    auto it = getCreatorMap().find(key);

    if (it == getCreatorMap().end())
        throw (std::invalid_argument("The creator you are trying to unregister was never registered.\n"));

    getCreatorMap().erase(key);
}

template<typename KeyType, typename ProductBase>
typename Factory<KeyType, ProductBase>::s_ptr_t Factory<KeyType, ProductBase>::
getCreator(const KeyType& key)
{
    auto it = getCreatorMap().find(key);

    if (it == getCreatorMap().end())
        throw (std::invalid_argument("The creator you are trying to access was never registered.\n"));

    return it->second;
}

template<typename KeyType, typename ProductBase>
std::unique_ptr<ProductBase> Factory<KeyType, ProductBase>::create(const KeyType& key)
{
    return getCreator(key) -> create();
}

}           // namespace util
}           // namespace lstr

#endif      // end include guard
