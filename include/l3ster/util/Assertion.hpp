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
template < std::integral I >
constexpr auto toString(I i) -> std::string
{
    constexpr auto max_int_str_len = 40;
    auto           buf             = std::array< char, max_int_str_len >{};
    const auto     result          = std::to_chars(buf.data(), buf.data() + max_int_str_len, i);
    return result.ec == std::errc{} ? std::string{buf.data(), result.ptr} : std::string{"[format error]"};
}

inline constexpr auto makeErrMsg(std::string_view err_message, std::source_location src_loc) -> std::string
{
    using namespace std::string_literals;
    using namespace std::string_view_literals;
    if consteval
    {
        auto retval = "At:          "s;
        retval += src_loc.file_name();
        retval += ':';
        retval += toString(src_loc.line());
        retval += ':';
        retval += toString(src_loc.column());
        retval += "\nIn function: ";
        retval += src_loc.function_name();
        retval += "\nAssertion failed";
        retval += err_message.empty() ? ""sv : ": ";
        retval += err_message.empty() ? ""sv : err_message;
        retval += '\n';
        return retval;
    }
    else
    {
        return std::format("At:          {}:{}:{}\nIn function: {}\nAssertion failed{}{}\n",
                           src_loc.file_name(),
                           src_loc.line(),
                           src_loc.column(),
                           src_loc.function_name(),
                           err_message.empty() ? ""sv : ": ",
                           err_message.empty() ? ""sv : err_message);
    }
}

template < typename Exception >
constexpr void throwErrorWithLocation(std::string_view err_message, std::source_location src_loc)
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
constexpr void throwingAssert(bool                 condition,
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
