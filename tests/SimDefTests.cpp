#include "l3ster/simdef/Simulation.hpp"

#include "catch2/catch.hpp"

using namespace lstr;

static consteval auto makeSim()
{
    using namespace std::string_view_literals;
    const auto dummy_kernel1 = [](auto in) {
    };
    const auto dummy_kernel2 = [](auto in) {
    };
    const auto dummy_kernel3 = [](auto in) {
    };
    const auto dummy_kernel4 = [](auto in) {
    };
    const auto dummy_kernel5 = [](auto in) {
    };

    def::Simulation sim{def::Kernel{"k1", dummy_kernel1},
                        def::Kernel{"k2", dummy_kernel2},
                        def::Kernel{"k3", dummy_kernel3},
                        def::Kernel{"k4", dummy_kernel4},
                        def::Kernel{"k5", dummy_kernel5}};
    const auto      token1 = sim.getKernelToken("k1");
    const auto      token2 = sim.getKernelToken("k2");
    const auto      token3 = sim.getKernelToken("k3");
    const auto      token4 = sim.getKernelToken("k4");
    const auto      token5 = sim.getKernelToken("k5");
    const auto      field  = sim.components().addField("field", 3, {1, 2});
    sim.components().addValue("val", 1);
    auto eq  = sim.components().addEquation(token1, {field}, {0}, 1, 0);
    auto bc  = sim.components().addBoundaryCondition(token2, {field}, {1, 2, 3, 4}, 1, 0);
    auto dbc = sim.components().addDirichletBoundaryCondition(token5, {field}, {5, 6});
    sim.components().addDomainTransform(token3);
    sim.components().addDomainReduction(token4);

    sim.defineProblem({eq}, {bc}, {dbc});

    return sim;
}

static consteval void checkFields(const auto& sim)
{
    const auto& field = sim.components().getFields().front();
    if (field.name != "field")
        throw "Incorrect field name";
    if (field.n_components != 3)
        throw "Incorrect number of components";
    if (field.domains.size() != 2 or field.domains[0] != 1 or field.domains[1] != 2)
        throw "Incorrect domains";
}
static consteval void checkValues(const auto& sim)
{
    const auto& value = sim.components().getValues().front();
    if (value.name != "val")
        throw "Incorrect value name";
    if (value.n_components != 1)
        throw "Incorrect number of components";
}
static consteval void checkEquations(const auto& sim)
{
    if (const auto& eq = sim.components().getEquations().front();
        eq.fields.size() != 1 or eq.fields[0] != std::addressof(sim.components().getFields().front()))
        throw "Equation defined over incorrect fields";
    if (const auto& bc = sim.components().getBoundaryConditions().front();
        bc.fields.size() != 1 or bc.fields[0] != std::addressof(sim.components().getFields().front()))
        throw "Boundary condition defined over incorrect fields";
    if (const auto& dbc = sim.components().getDirichletConditions().front();
        dbc.fields.size() != 1 or dbc.fields[0] != std::addressof(sim.components().getFields().front()))
        throw "Dirichlet boundary condition defined over incorrect fields";
}
static consteval void checkOps(const auto& sim)
{
    if (sim.components().getDomainTransforms().size() != 1)
        throw "Incorrect number of domain transforms";
    if (sim.components().getBoundaryTransforms().size() != 0)
        throw "Incorrect number of boundary transforms";
    if (sim.components().getDomainReductions().size() != 1)
        throw "Incorrect number of domain reductions";
    if (sim.components().getBoundaryReductions().size() != 0)
        throw "Incorrect number of boundary reductions";
}
static consteval bool checkSim()
{
    const auto sim = makeSim();
    checkFields(sim);
    checkValues(sim);
    checkEquations(sim);
    checkOps(sim);
    return true;
}

TEST_CASE("Adding fields", "[simdef]")
{
    REQUIRE(checkSim());
}
