#include "l3ster/simstructure/SimulationDef.hpp"

#include "catch2/catch.hpp"

template < typename T >
constexpr bool always_true_v = true;

TEST_CASE("Simulation definition", "[def]")
{
    using namespace lstr::def;

    SECTION("Advection")
    {
        constexpr auto make_sim = [] {
            constexpr auto advection_kernel_2D = [](const auto& in, auto& out) {
                std::array adv_velocity{1., 1.};

                out.At(0, 0) = 1.;
                out.Ax(0, 0) = adv_velocity[0];
                out.Ay(0, 0) = adv_velocity[1];
            };

            constexpr Mesh mesh{"mesh_file.msh"};

            constexpr double          T_start = 0., T_end = 10.;
            static constexpr Timeline timeline{T_start, T_end};

            constexpr Field< Space::D2, Time::Transient > phi{&timeline, 0.};
            constexpr FieldSet                            fields{phi};

            Support sup_phi{1};
            Support top{2};
            Support bot{3};
            Support left{4};
            Support right{5};

            TimeSolver< TimeSchemes::BDF2 > time_solver;
            time_solver.dt = 1e-2;

            Equation       advection_eq{advection_kernel_2D, sup_phi, fields, time_solver};
            DirichletBC    top_bc{phi, top, 0.};
            DirichletBC    bot_bc{phi, bot, 0.};
            DirichletBC    left_bc{phi, left, 1.};
            DirichletBC    right_bc{phi, right, 0.};
            DirichletBCSet bcs{top_bc, bot_bc, left_bc, right_bc};
            Physics        advection{advection_eq, EquationSet{}, bcs};

            AlgebraicSolver< AlgebraicSolverTypes::DirectSuperLU > alg_solver;
            System                                                 advection_system{alg_solver, advection};
            Subiter                                                advection_subiter{advection_system};
            Problem                                                advection_problem{advection_subiter};
            Simulation                                             simulation{advection_problem, mesh};

            return simulation;
        };
        constexpr auto sim = make_sim();

        REQUIRE(always_true_v< decltype(sim) >);
    }
}