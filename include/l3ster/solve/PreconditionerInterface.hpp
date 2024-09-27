#ifndef L3STER_SOLVE_PRECONDITIONERINTERFACE_HPP
#define L3STER_SOLVE_PRECONDITIONERINTERFACE_HPP

#include "l3ster/common/TrilinosTypedefs.h"

#include <functional>

namespace lstr::solvers
{
namespace detail
{
template < typename Precond >
concept PreconditionerBase_c = requires {
    typename Precond::Options;
    requires std::default_initializable< typename Precond::Options >;
    typename Precond::Options::Preconditioner;
    requires std::same_as< typename Precond::Options::Preconditioner, Precond >;
};
} // namespace detail

template < typename Precond >
concept OperatorBasedPreconditioner_c =
    detail::PreconditionerBase_c< Precond > and
    requires(typename Precond::Options const& opts, Teuchos::RCP< const tpetra_operator_t > const& op) {
        Precond::create(opts, op);
        requires std::convertible_to< decltype(Precond::create(opts, op)), Teuchos::RCP< tpetra_operator_t > >;
    };

template < typename Precond >
concept MatrixBasedPreconditioner_c =
    detail::PreconditionerBase_c< Precond > and
    requires(typename Precond::Options const& opts, Teuchos::RCP< const tpetra_crsmatrix_t > const& matrix) {
        { Precond::create(opts, matrix) } -> std::convertible_to< Teuchos::RCP< tpetra_operator_t > >;
    };

template < typename Precond >
concept Preconditioner_c = MatrixBasedPreconditioner_c< Precond > or OperatorBasedPreconditioner_c< Precond >;

template < typename PrecondOpts >
concept PreconditionerOptions_c = requires {
    typename PrecondOpts::Preconditioner;
    requires Preconditioner_c< typename PrecondOpts::Preconditioner >;
};

struct NullPreconditioner
{
    struct Options
    {
        using Preconditioner = NullPreconditioner;
    };
    static auto create(const Options&,
                       const Teuchos::RCP< const tpetra_operator_t >&) -> Teuchos::RCP< tpetra_operator_t >
    {
        return {};
    }
};

namespace detail
{
template < PreconditionerOptions_c Options >
auto initPreconditioner(const Options&                                 opts,
                        const Teuchos::RCP< const tpetra_operator_t >& op) -> Teuchos::RCP< tpetra_operator_t >
{
    using Preconditioner = Options::Preconditioner;
    if constexpr (MatrixBasedPreconditioner_c< Preconditioner >)
    {
        if (const auto as_matrix = Teuchos::rcp_dynamic_cast< const tpetra_crsmatrix_t >(op, false); as_matrix)
            return Preconditioner::create(opts, as_matrix);
    }
    if constexpr (OperatorBasedPreconditioner_c< Preconditioner >)
        return Preconditioner::create(opts, op);
    std::unreachable();
}
} // namespace detail

class DeferredPreconditionerInitializer
{
public:
    template < PreconditionerOptions_c Options >
    DeferredPreconditionerInitializer(const Options& opts)
        : m_init{[opts](const Teuchos::RCP< const tpetra_operator_t >& op) {
              return detail::initPreconditioner(opts, op);
          }}
    {}
    auto operator()(const Teuchos::RCP< const tpetra_operator_t >& op) const { return m_init(op); }

private:
    std::function< Teuchos::RCP< tpetra_operator_t >(const Teuchos::RCP< const tpetra_operator_t >&) > m_init;
};
} // namespace lstr::solvers
#endif // L3STER_SOLVE_PRECONDITIONERINTERFACE_HPP
