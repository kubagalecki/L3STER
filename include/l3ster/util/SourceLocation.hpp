#ifndef L3STER_SOURCELOCATION_HPP
#define L3STER_SOURCELOCATION_HPP

#include <algorithm>
#include <array>
#include <charconv>
#include <concepts>
#include <iterator>
#include <ranges>
#include <source_location>
#include <span>
#include <stdexcept>
#include <string>

namespace lstr::util
{
[[nodiscard]] inline auto parseSourceLocation(std::source_location src_loc) -> std::string
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
    const auto in_fun_txt = "; In function "sv;
    const auto line_txt   = parse_int(line_buf, src_loc.line());
    const auto col_txt    = parse_int(col_buf, src_loc.line());

    auto retval = std::string{};
    retval.reserve(file_name.size() + fun_name.size() + in_fun_txt.size() + line_txt.size() + col_txt.size() + 3u);
    std::ranges::copy(file_name, std::back_inserter(retval));
    retval.push_back(':');
    std::ranges::copy(line_txt, std::back_inserter(retval));
    retval.push_back(':');
    std::ranges::copy(col_txt, std::back_inserter(retval));
    std::ranges::copy(in_fun_txt, std::back_inserter(retval));
    std::ranges::copy(fun_name, std::back_inserter(retval));
    retval.push_back('\n');
    return retval;
}

inline auto runtimeError(std::string_view err_messge, std::source_location src_loc = std::source_location::current())
{
    auto err_msg_src_located = parseSourceLocation(src_loc);
    err_msg_src_located.reserve(err_msg_src_located.size() + err_messge.size());
    std::ranges::copy(err_messge, std::back_inserter(err_msg_src_located));
    throw std::runtime_error{err_msg_src_located.c_str()};
}
} // namespace lstr::util
#endif // L3STER_SOURCELOCATION_HPP
