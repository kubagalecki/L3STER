#ifndef L3STER_UTIL_ASSERTION_HPP
#define L3STER_UTIL_ASSERTION_HPP

#include <algorithm>
#include <array>
#include <charconv>
#include <concepts>
#include <exception>
#include <format>
#include <functional>
#include <iostream>
#include <iterator>
#include <ranges>
#include <source_location>
#include <span>
#include <stdexcept>
#include <string>

namespace lstr::util
{
namespace detail
{
inline auto makeErrMsg(std::string_view err_message, std::source_location src_loc) -> std::string
{
    using namespace std::string_view_literals;
    return std::format("Assertion failed{}{}\nAt {}:{}:{}\nIn function {}\n",
                       err_message.empty() ? ""sv : ": ",
                       err_message.empty() ? ""sv : err_message,
                       src_loc.file_name(),
                       src_loc.line(),
                       src_loc.column(),
                       src_loc.function_name());
}

template < typename Exception >
void throwErrorWithLocation(std::string_view err_message, std::source_location src_loc)
{
    throw std::invoke([&] {
        if constexpr (std::constructible_from< Exception, const char* >)
        {
            const auto err_msg = makeErrMsg(err_message, src_loc);
            return Exception{err_msg.c_str()};
        }
        else if constexpr (std::default_initializable< Exception >)
            return Exception{};
        else
            static_assert(not std::same_as< Exception, Exception >,
                          "The exception must be either default-initializable or constructible from a const char*");
    });
}

inline void terminateWithMessage(std::string_view err_message, std::source_location src_loc)
{
    std::cerr << makeErrMsg(err_message, src_loc);
    std::terminate();
}
} // namespace detail

template < typename Exception = std::runtime_error >
void throwingAssert(bool                 condition,
                    std::string_view     err_msg = {},
                    std::source_location src_loc = std::source_location::current())
{
    if (not condition)
        detail::throwErrorWithLocation< Exception >(err_msg, src_loc);
}

inline void terminatingAssert(bool                 condition,
                              std::string_view     err_msg,
                              std::source_location src_loc = std::source_location::current())
{
    if (not condition)
        detail::terminateWithMessage(err_msg, src_loc);
}
} // namespace lstr::util
#endif // L3STER_UTIL_ASSERTION_HPP
