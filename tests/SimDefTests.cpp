#include "l3ster/simdef/Timeline.hpp"

#include "catch2/catch.hpp"

using namespace lstr;

static consteval bool checkAddField()
{
    using namespace std::string_view_literals;
    const auto dummy_kernel1 = [](auto in) {
    };
    const auto dummy_kernel2 = [](auto in) {
    };
    def::KernelSet kernels{def::Kernel{"k1"sv, dummy_kernel1}, def::Kernel{"k2"sv, dummy_kernel2}};
    def::Timeline  tl{kernels};
    const auto     vel = tl.addField("velocity", 3, {1, 2});
    return vel->name == "velocity" and vel->n_components == 3 and vel->domains[0] == 1 and vel->domains[1] == 2;
}

TEST_CASE("Adding fields", "[simdef]")
{
    static_assert(checkAddField());
}
