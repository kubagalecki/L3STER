#ifndef L3STER_UTIL_ASSERTION_HPP
#define L3STER_UTIL_ASSERTION_HPP

#include <algorithm>
#include <array>
#include <charconv>
#include <concepts>
#include <exception>
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
inline auto parseSourceLocation(std::source_location src_loc) -> std::string
{
    using namespace std::string_view_literals;
    constexpr std::size_t buf_size = 32;
    using buf_t = std::array< char, buf_size >; // assume length of formatted integer is less than buf_size
    buf_t          line_buf{}, col_buf{};
    constexpr auto parse_int = [](buf_t& buf, std::integral auto value) -> std::span< const char > {
        const auto [end, err] = std::to_chars(buf.data(), std::next(buf.data(), std::ranges::ssize(buf)), value);
        if (err != std::errc{})
            throw std::runtime_error{"Parsing source location failed"};
        return {buf.data(), end};
    };

    const auto file_name  = std::string_view{src_loc.file_name()};
    const auto fun_name   = std::string_view{src_loc.function_name()};
    const auto in_fun_txt = "\n\tIn function "sv;
    const auto line_txt   = parse_int(line_buf, src_loc.line());
    const auto col_txt    = parse_int(col_buf, src_loc.column());

    auto retval = std::string{"\t"};
    retval.reserve(file_name.size() + fun_name.size() + in_fun_txt.size() + line_txt.size() + col_txt.size() + 5u);
    std::ranges::copy(file_name, std::back_inserter(retval));
    retval.push_back(':');
    std::ranges::copy(line_txt, std::back_inserter(retval));
    retval.push_back(':');
    std::ranges::copy(col_txt, std::back_inserter(retval));
    std::ranges::copy(in_fun_txt, std::back_inserter(retval));
    std::ranges::copy(fun_name, std::back_inserter(retval));
    retval.append("\n\t");
    return retval;
}

inline auto makeErrMsg(std::string_view err_message, std::source_location src_loc) -> std::string
{
    auto err_msg_src_located = parseSourceLocation(src_loc);
    err_msg_src_located.reserve(err_msg_src_located.size() + err_message.size());
    std::ranges::copy(err_message, std::back_inserter(err_msg_src_located));
    return err_msg_src_located;
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
