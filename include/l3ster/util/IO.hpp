#ifndef L3STER_UTIL_IO_HPP
#define L3STER_UTIL_IO_HPP

#include "l3ster/util/Assertion.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <filesystem>
#include <fstream>
#include <ranges>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace lstr::util
{
template < std::default_initializable T >
T extract(std::istream& istream)
{
    T retval;
    istream >> retval;
    return retval;
}

template < std::default_initializable T, size_t N >
auto extract(std::istream& istream) -> std::array< T, N >
{
    auto retval = std::array< T, N >{};
    for (auto& v : retval)
        istream >> v;
    return retval;
}

template < std::default_initializable T >
void ignore(std::istream& istream, size_t n = 1)
{
    T dummy;
    for (size_t i = 0; i != n; ++i)
        istream >> dummy;
}

class MmappedFile
{
public:
    MmappedFile() = default;
    inline MmappedFile(const std::filesystem::path& filename);

    [[nodiscard]] auto   view() const -> std::string_view { return {get(), size()}; }
    [[nodiscard]] char*  get() const { return m_data.get(); }
    [[nodiscard]] size_t size() const { return m_data.get_deleter().size; }

private:
    struct MunmapDeleter
    {
        MunmapDeleter() noexcept = default;
        MunmapDeleter(size_t size_) : size(size_) {}
        void operator()(char* ptr) const { munmap(ptr, size); }

        size_t size{};
    };

    std::unique_ptr< char, MunmapDeleter > m_data;
};

MmappedFile::MmappedFile(const std::filesystem::path& filename)
{
    auto open_mode = O_RDONLY;
#ifdef _LARGEFILE64_SOURCE
    open_mode |= O_LARGEFILE;
#endif
    const int file = open(filename.c_str(), open_mode);
    throwingAssert(file != -1, "Failed to mmap file");
    struct stat file_stats;
    throwingAssert(!fstat(file, &file_stats), "Could not read file info via fstat syscall");
    const auto size = static_cast< size_t >(file_stats.st_size);
    void*      addr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, file, 0);
    throwingAssert(addr != MAP_FAILED, "Failed to mmap file");
    m_data = std::unique_ptr< char, MunmapDeleter >{static_cast< char* >(addr), MunmapDeleter{size}};
    throwingAssert(!close(file), "Failed to close file after mmap");
}

class MmappedStreambuf : public std::basic_streambuf< char >
{
public:
    explicit MmappedStreambuf(MmappedFile file) : m_file(std::move(file))
    {
        setg(m_file.get(), m_file.get(), std::next(m_file.get(), m_file.size()));
    }
    bool skipPast(std::string_view search_str)
    {
        const auto remaining   = std::string_view{gptr(), egptr()};
        const auto found_range = std::ranges::search(remaining, search_str);
        setg(eback(), const_cast< char* >(found_range.end()), egptr());
        return std::ranges::begin(found_range) != remaining.end();
    }

protected:
    int_type underflow() override { return gptr() == egptr() ? traits_type::eof() : traits_type::to_int_type(*gptr()); }
    std::streamsize showmanyc() override { return static_cast< std::streamsize >(std::distance(gptr(), egptr())); }

private:
    MmappedFile m_file;
};
} // namespace lstr::util
#endif // L3STER_UTIL_IO_HPP
