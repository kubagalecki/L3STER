#ifndef L3STER_SOLVE_PRECONDITIONER_MANAGER_HPP
#define L3STER_SOLVE_PRECONDITIONER_MANAGER_HPP

#include "l3ster/common/TrilinosTypedefs.h"
#include "l3ster/solve/Preconditioners.hpp"
#include "l3ster/util/Meta.hpp"

#include <algorithm>
#include <array>
#include <string_view>
#include <variant>

namespace lstr::solvers
{
enum class PrecondType
{
    None       = 0,
    Richardson = 1,
    Jacobi     = 2,
    SGS        = 3,
    Chebyshev  = 4,

    // Update Count when adding new preconditioner type
    Count = 5
};
namespace detail
{
inline constexpr auto precond_types = std::invoke([] {
    auto retval = std::array< PrecondType, static_cast< size_t >(PrecondType::Count) >{};
    std::ranges::generate(retval, [i = 0]() mutable { return static_cast< PrecondType >(i++); });
    return retval;
});
using precond_types_pack_t          = util::convert_array_to_value_pack_t< precond_types >;
using precond_types_for_deduction_t = util::split_value_pack_t< precond_types_pack_t >;

template < PrecondType >
inline constexpr std::string_view preconditioner_name{"None"};
} // namespace detail

template < PrecondType >
struct PrecondOpts
{};

// Note: Can't do designated initialization with a base class, so we really do need to repeat the members below
template <>
struct PrecondOpts< PrecondType::Richardson >
{
    int   sweeps         = 1;
    val_t damping        = 1.;
    bool  enable_l1      = false;
    val_t l1_eta         = 1.5;
    val_t diag_threshold = 0.;
};

template <>
struct PrecondOpts< PrecondType::Jacobi >
{
    int   sweeps         = 1;
    val_t damping        = 1.;
    bool  enable_l1      = false;
    val_t l1_eta         = 1.5;
    val_t diag_threshold = 0.;
};

template <>
struct PrecondOpts< PrecondType::SGS >
{
    int   sweeps         = 1;
    val_t damping        = 1.;
    bool  enable_l1      = false;
    val_t l1_eta         = 1.5;
    val_t diag_threshold = 0.;
};

template <>
struct PrecondOpts< PrecondType::Chebyshev >
{
    int   degree          = 1;   // Chebyshev polynomial degree
    val_t cond_est        = 30.; // Estimated ratio of max/min eigenvalues
    int   max_power_iters = 10;  // Max number of iterations of power method for estimating max eigenvalue
    val_t boost_factor    = 1.1; // Factor for max eigenvalue estimation to ensure margin
    val_t diag_threshold  = 0.;  // Diagonal entries are bounded from below by this value
};

namespace detail
{
template <>
inline constexpr std::string_view preconditioner_name< PrecondType::Richardson >{"RELAXATION"};
template <>
inline constexpr std::string_view preconditioner_name< PrecondType::Jacobi >{"RELAXATION"};
template <>
inline constexpr std::string_view preconditioner_name< PrecondType::SGS >{"RELAXATION"};
template <>
inline constexpr std::string_view preconditioner_name< PrecondType::Chebyshev >{"CHEBYSHEV"};
} // namespace detail

class PreconditionerManager
{
    using precond_opt_var_t = util::apply_in_out_t< PrecondOpts, std::variant, detail::precond_types_for_deduction_t >;

public:
    template < PrecondType type >
    explicit PreconditionerManager(const PrecondOpts< type >& opts) : m_options{opts}
    {}

    void initialize(const Teuchos::RCP< const tpetra_operator_t >& op)
    {
        assertValidConfig(op);
        std::visit([this, &op](const auto& opts) { initImpl(op, opts); }, m_options);
    }
    void compute()
    {
        if (m_precond_op.is_null())
            return;
        computeImpl();
    }

    auto get() const -> Teuchos::RCP< const tpetra_operator_t > { return m_precond_op; }

private:
    inline auto getType() const -> PrecondType;
    inline void assertValidConfig(const Teuchos::RCP< const tpetra_operator_t >& op) const;
    template < PrecondType type >
    void initImpl(const Teuchos::RCP< const tpetra_operator_t >& op, const PrecondOpts< type >& options);
    template < PrecondType type >
    auto initIfpack2Impl(const Teuchos::RCP< const tpetra_crsmatrix_t >& matrix,
                         const PrecondOpts< type >&                      options) -> Teuchos::RCP< tpetra_operator_t >;
    template < PrecondType type >
    auto        initNativeImpl(const Teuchos::RCP< const tpetra_operator_t >& op,
                               const PrecondOpts< type >&                     opts) -> Teuchos::RCP< tpetra_operator_t >;
    inline void computeImpl();

    template < PrecondType type >
    static auto parsmakeIfpack2Params(const PrecondOpts< type >& opts) -> Teuchos::ParameterList;

    precond_opt_var_t                 m_options;
    Teuchos::RCP< tpetra_operator_t > m_precond_op;
};

void PreconditionerManager::assertValidConfig(const Teuchos::RCP< const tpetra_operator_t >& op) const
{
    if (getType() == PrecondType::None or getType() == PrecondType::Richardson)
        return; // Available for all operators
    if (auto as_crsmat = Teuchos::rcp_dynamic_cast< const tpetra_crsmatrix_t >(op, false); as_crsmat)
        return; // All preconditioners are available if the underlying operator is a Tpetra::CrsMatrix
    util::throwingAssert(
        getType() != PrecondType::Chebyshev,
        "The Chebyshev preconditioner is available only for operators which are backed by a CRS matrix");
    util::throwingAssert(
        getType() == PrecondType::Jacobi and op->hasDiagonal(),
        "The Jacobi preconditioner is available only for matrix-free operators which expose the diagonal");
}

void PreconditionerManager::computeImpl()
{
    const auto try_compute_as = [&]< typename T >(util::TypePack< T >) {
        const auto downcast_ptr = Teuchos::rcp_dynamic_cast< T >(m_precond_op, false);
        const auto success      = not downcast_ptr.is_null();
        if (success)
            downcast_ptr->compute();
        return success;
    };
    constexpr auto ifpack2_precond = util::TypePack< Ifpack2::Preconditioner<> >{};
    constexpr auto native_precond  = util::TypePack< PreconditionerBase >{};
    if (try_compute_as(ifpack2_precond))
        return;
    if (try_compute_as(native_precond))
        return;
}

auto PreconditionerManager::getType() const -> PrecondType
{
    return std::visit< PrecondType >([]< PrecondType type >(PrecondOpts< type >) { return type; }, m_options);
}

template < PrecondType type >
auto PreconditionerManager::parsmakeIfpack2Params(const PrecondOpts< type >& opts) -> Teuchos::ParameterList
{
    auto retval = Teuchos::ParameterList{};
    if constexpr (detail::preconditioner_name< type > == "RELAXATION")
    {
        constexpr auto get_relax_name = [] {
            if constexpr (type == PrecondType::Richardson)
                return "Richardson";
            else if constexpr (type == PrecondType::Jacobi)
                return "Jacobi";
            else
                return "Symmetric Gauss-Seidel";
        };

        retval.set("relaxation: type", get_relax_name());
        retval.set("relaxation: sweeps", opts.sweeps);
        retval.set("relaxation: damping factor", opts.damping);
        retval.set("relaxation: use l1", opts.enable_l1);
        retval.set("relaxation: l1 eta", opts.l1_eta);
        retval.set("relaxation: fix tiny diagonal entries", opts.diag_threshold != 0.);
        retval.set("relaxation: min diagonal value", opts.diag_threshold);
    }
    else if constexpr (type == PrecondType::Chebyshev)
    {
        retval.set("chebyshev: degree", opts.degree);
        retval.set("chebyshev: ratio eigenvalue", opts.cond_est);
        retval.set("chebyshev: eigenvalue max iterations", opts.max_power_iters);
        retval.set("chebyshev: min diagonal value", opts.diag_threshold);
        retval.set("chebyshev: boost factor", opts.boost_factor);
    }
    return retval;
}

template < PrecondType type >
auto PreconditionerManager::initNativeImpl(const Teuchos::RCP< const tpetra_operator_t >& op,
                                           const PrecondOpts< type >& opts) -> Teuchos::RCP< tpetra_operator_t >
{
    if constexpr (type == PrecondType::Richardson)
        return util::makeTeuchosRCP< RichardsonPreconditioner >(op, opts.damping, opts.sweeps);
    else if constexpr (type == PrecondType::Jacobi)
        return util::makeTeuchosRCP< JacobiPreconditioner >(op, opts.damping, opts.diag_threshold, opts.sweeps);
    else
        return {};
}

template < PrecondType type >
auto PreconditionerManager::initIfpack2Impl(const Teuchos::RCP< const tpetra_crsmatrix_t >& matrix,
                                            const PrecondOpts< type >& options) -> Teuchos::RCP< tpetra_operator_t >
{
    const auto precond_name = std::string{detail::preconditioner_name< type >};
    auto       retval       = Ifpack2::Factory::create(precond_name, matrix);
    const auto params       = parsmakeIfpack2Params(options);
    retval->setParameters(params);
    retval->initialize();
    util::throwingAssert(retval->isInitialized(), "Failed to initialize Ifpack2 preconditioner");
    return retval;
}

template < PrecondType type >
void PreconditionerManager::initImpl(const Teuchos::RCP< const tpetra_operator_t >& op,
                                     const PrecondOpts< type >&                     options)
{
    if constexpr (type == PrecondType::None)
        return;
    if (const auto as_crsmatrix = Teuchos::rcp_dynamic_cast< const tpetra_crsmatrix_t >(op, false); as_crsmatrix)
        m_precond_op = initIfpack2Impl(as_crsmatrix, options);
    else
        m_precond_op = initNativeImpl(op, options);
}
} // namespace lstr::solvers

namespace lstr
{
using RichardsonOpts = solvers::PrecondOpts< solvers::PrecondType::Richardson >;
using JacobiOpts     = solvers::PrecondOpts< solvers::PrecondType::Jacobi >;
using SGSOpts        = solvers::PrecondOpts< solvers::PrecondType::SGS >;
using ChebyshevOpts  = solvers::PrecondOpts< solvers::PrecondType::Chebyshev >;
} // namespace lstr
#endif // L3STER_SOLVE_PRECONDITIONER_MANAGER_HPP
