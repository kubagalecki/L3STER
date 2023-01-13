#include "l3ster/simdef/Simulation.hpp"

#include "catch2/catch.hpp"

using namespace lstr;

static consteval auto makeSim()
{
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

    const auto field = sim.components().defineField("field", 3, {1, 2});
    const auto eq    = sim.components().defineEquation("k1", {field}, {0}, 1, 0);
    const auto bc    = sim.components().defineBoundaryCondition("k2", {field}, {1, 2, 3, 4}, 1, 0);
    const auto dbc   = sim.components().defineDirichletBoundaryCondition("k5", {field}, {5, 6});

    std::ignore = sim.components().defineValue("val", 1);
    std::ignore = sim.components().defineDomainTransform("k3");
    std::ignore = sim.components().defineDomainReduction("k4");

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
