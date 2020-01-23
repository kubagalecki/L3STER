// Implementation of the visitor design pattern

#ifndef L3STER_INCGUARD_UTIL_VISITOR_HPP
#define L3STER_INCGUARD_UTIL_VISITOR_HPP

#include <utility>
#include <tuple>
#include <memory>

namespace lstr::util
{

//////////////////////////////////////////////////////////////////////////////////////////////
//                                      VISITOR BASE CLASS                                  //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
Base class for Visitor, templated with a parameter pack representing the visitable types
*/
template <typename ... Visitables>
class VisitorBase;

template <typename V>
class VisitorBase<V>
{
public:
    virtual void visit(V&)          const = 0;
    virtual void cvisit(const V&)   const = 0;
    virtual ~VisitorBase()          = default;
};

template <typename V, typename... Visitables>
class VisitorBase<V, Visitables ...> : public VisitorBase<Visitables ...>
{
public:
    using VisitorBase<Visitables ...>::visit;
    using VisitorBase<Visitables ...>::cvisit;

    virtual void visit(V&)          const = 0;
    virtual void cvisit(const V&)   const = 0;
    virtual ~VisitorBase()          = default;
};

//////////////////////////////////////////////////////////////////////////////////////////////
//                                        VISITOR CLASS                                     //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
The visitor class stores a function object and has 'visit' and 'cvisit' member functions,
which can be called to visit a (possibly const) object of type belonging to the parameter
pack the base VisitorBase class is templated with
*/
template <typename ... Visitables>
class Visitor;

template <typename FunctorType, typename ... Visitables, typename V>
class Visitor<FunctorType, std::tuple<Visitables ...>, V> :
    public VisitorBase<Visitables ...>
{
public:
    Visitor(FunctorType&& f) : func{std::forward<FunctorType>(f)} {}

    template <typename ... Types>
    Visitor(Types&& ... Args) : func{std::forward<Types>(Args) ...} {}

    using VisitorBase<Visitables ...>::visit;
    using VisitorBase<Visitables ...>::cvisit;

    void cvisit(const V& o) const final
    {
        func(o);
    }

    void visit(V& o) const final
    {
        func(o);
    }

protected:
    mutable FunctorType func;   // the function object can have a mutable internal state
};

template <typename FunctorType, typename ... Visitables, typename VC, typename ... V>
class Visitor<FunctorType, std::tuple<Visitables ...>, VC, V...> :
    public Visitor<FunctorType, std::tuple<Visitables ...>, V...>
{
public:
    Visitor(FunctorType&& f) :
        Visitor<FunctorType, std::tuple<Visitables ...>, V...>
    {
        std::forward<FunctorType>(f)
    } {}

    template <typename ... Types>
    Visitor(Types&& ... Args) :
        Visitor<FunctorType, std::tuple<Visitables ...>, V...>
    {
        std::forward<Types>(Args) ...
    } {}

    using Visitor<FunctorType, std::tuple<Visitables ...>, V...>::visit;
    using Visitor<FunctorType, std::tuple<Visitables ...>, V...>::cvisit;
    using Visitor<FunctorType, std::tuple<Visitables ...>, V...>::func;

    void cvisit(const VC& o) const final
    {
        func(o);
    }

    void visit(VC& o) const final
    {
        func(o);
    }
};

// Factory function for the Visitor class, effectively constitutes its interface
template <typename ... Types, typename V>
std::unique_ptr< VisitorBase<Types ...> > make_visitor(V&& v_obj)
{
    return std::make_unique < Visitor< V, std::tuple<Types ...>, Types ... > >
           (std::forward<V>(v_obj));
}

//////////////////////////////////////////////////////////////////////////////////////////////
//                                     VISITABLE BASE CLASS                                 //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
Base class for Visitable, templated with a parameter pack representing the visitable types
*/
template <typename ... Visitables>
class VisitableBase
{
public:
    void caccept(VisitorBase<Visitables ...>&& vb)  = 0;
    void accept(VisitorBase<Visitables ...>&& vb)   = 0;
};

}           // namespace lstr::util

#endif      // end include guard
