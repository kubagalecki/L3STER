#ifndef L3STER_ALGSYS_SUMFACTORIZATION_HPP
#define L3STER_ALGSYS_SUMFACTORIZATION_HPP

#include "l3ster/algsys/EvaluateLocalOperator.hpp"
#include "l3ster/mesh/LocalMeshView.hpp"
#include "l3ster/post/FieldAccess.hpp"
#include "l3ster/util/Simd.hpp"

namespace lstr::algsys
{
struct BasisParams
{
    el_o_t               basis_order;
    q_o_t                quad_order;
    basis::BasisType     basis_type;
    quad::QuadratureType quad_type;

    consteval int n_bases1d() const { return basis_order + 1; }
    consteval int n_qps1d() const { return static_cast< int >(quad::getRefQuadSize(quad_type, quad_order)); }
};

namespace detail
{
template < BasisParams basis_params >
auto makeInterpolationMatrix()
{
    const auto& [qps, _]    = quad::getReferenceQuadrature< basis_params.quad_type, basis_params.quad_order >();
    const auto basis_at_qps = basis::detail::
        evalRefBasisAtPoints< basis_params.basis_type, mesh::ElementType::Line, basis_params.basis_order >(qps);
    auto retval = util::eigen::RowMajorMatrix< val_t, basis_params.n_bases1d(), basis_params.n_qps1d() >{};
    for (q_o_t qi = 0; qi != basis_params.n_qps1d(); ++qi)
        for (el_o_t bi = 0; bi != basis_params.n_bases1d(); ++bi)
            retval(bi, qi) = basis_at_qps.values[qi][bi];
    return retval;
}

template < BasisParams basis_params >
auto makeDerivativeMatrix()
{
    const auto& [qps, _]    = quad::getReferenceQuadrature< basis_params.quad_type, basis_params.quad_order >();
    const auto basis_at_qps = basis::detail::
        evalRefBasisAtPoints< basis_params.basis_type, mesh::ElementType::Line, basis_params.basis_order >(qps);
    auto retval = util::eigen::RowMajorMatrix< val_t, basis_params.n_bases1d(), basis_params.n_qps1d() >{};
    for (q_o_t qi = 0; qi != basis_params.n_qps1d(); ++qi)
        for (el_o_t bi = 0; bi != basis_params.n_bases1d(); ++bi)
            retval(bi, qi) = basis_at_qps.derivatives[qi][bi];
    return retval;
}

template < typename Matrix >
auto makeTransposed(const Matrix& matrix)
{
    using return_t = Eigen::Matrix< val_t, Matrix::ColsAtCompileTime, Matrix::RowsAtCompileTime, Eigen::RowMajor >;
    return return_t{matrix.transpose()};
}

template < BasisParams basis_params >
inline const auto interpolation_matrix = makeInterpolationMatrix< basis_params >();
template < BasisParams basis_params >
inline const auto derivative_matrix = makeDerivativeMatrix< basis_params >();
template < BasisParams basis_params >
inline const auto interpolation_matrix_trans = makeTransposed(interpolation_matrix< basis_params >);
template < BasisParams basis_params >
inline const auto derivative_matrix_trans = makeTransposed(derivative_matrix< basis_params >);

template < BasisParams basis_params, typename EigenMapIn, typename EigenMapOut >
void sumFactSweepBackInterp(EigenMapIn&& in, EigenMapOut&& out)
{
    out.noalias() = in.transpose() * interpolation_matrix< basis_params >;
}
template < BasisParams basis_params, typename EigenMapIn, typename EigenMapOut >
void sumFactSweepBackDer(EigenMapIn&& in, EigenMapOut&& out)
{
    out.noalias() = in.transpose() * derivative_matrix< basis_params >;
}
template < BasisParams basis_params, typename EigenMapIn, typename EigenMapOut >
void sumFactSweepForwardInterpAssign(EigenMapIn&& in, EigenMapOut&& out)
{
    out.noalias() = in.transpose() * interpolation_matrix_trans< basis_params >;
}
template < BasisParams basis_params, typename EigenMapIn, typename EigenMapOut >
void sumFactSweepForwardDerAccumulate(EigenMapIn&& in, EigenMapOut&& out)
{
    out.noalias() += in.transpose() * derivative_matrix_trans< basis_params >;
}

template < int ret_rows, int ret_cols, int rows, int cols, int... params >
auto makePsiPlusImpl(const Eigen::Matrix< val_t, rows, cols, params... >& mat)
{
    auto retval = util::eigen::RowMajorMatrix< val_t, ret_rows, ret_cols >{};
    for (int r = 0; r != rows / 2; ++r)
        for (int c = 0; c != ret_cols; ++c)
            retval(r, c) = mat(r, c) + mat(rows - r - 1, c);
    if constexpr (rows % 2)
        retval.template bottomRows< 1 >() = mat.template block< 1, ret_cols >(rows / 2, 0);
    return retval;
}

template < int ret_rows, int ret_cols, int rows, int cols, int... params >
auto makePsiMinusImpl(const Eigen::Matrix< val_t, rows, cols, params... >& mat)
{
    auto retval = util::eigen::RowMajorMatrix< val_t, ret_rows, ret_cols >{};
    for (int r = 0; r != ret_rows; ++r)
        for (int c = 0; c != ret_cols; ++c)
            retval(r, c) = mat(r, c) - mat(rows - r - 1, c);
    return retval;
}

template < int rows, int cols, int... params >
auto makePsiPlusInterp(const Eigen::Matrix< val_t, rows, cols, params... >& mat)
{
    constexpr int ret_rows = (rows + 1) / 2;
    constexpr int ret_cols = (cols + 1) / 2;
    return makePsiPlusImpl< ret_rows, ret_cols >(mat);
}

template < int rows, int cols, int... params >
auto makePsiMinusInterp(const Eigen::Matrix< val_t, rows, cols, params... >& mat)
{
    constexpr int ret_rows = rows / 2;
    constexpr int ret_cols = cols / 2;
    return makePsiMinusImpl< ret_rows, ret_cols >(mat);
}

template < int rows, int cols, int... params >
auto makePsiPlusDer(const Eigen::Matrix< val_t, rows, cols, params... >& mat)
{
    constexpr int ret_rows = (rows + 1) / 2;
    constexpr int ret_cols = cols / 2;
    return makePsiPlusImpl< ret_rows, ret_cols >(mat);
}

template < int rows, int cols, int... params >
auto makePsiMinusDer(const Eigen::Matrix< val_t, rows, cols, params... >& mat)
{
    constexpr int ret_rows = rows / 2;
    constexpr int ret_cols = (cols + 1) / 2;
    return makePsiMinusImpl< ret_rows, ret_cols >(mat);
}

template < BasisParams basis_params >
inline const auto psi_plus_interp_back = makePsiPlusInterp(interpolation_matrix< basis_params >);
template < BasisParams basis_params >
inline const auto psi_minus_interp_back = makePsiMinusInterp(interpolation_matrix< basis_params >);
template < BasisParams basis_params >
inline const auto psi_plus_interp_forward = makePsiPlusInterp(interpolation_matrix_trans< basis_params >);
template < BasisParams basis_params >
inline const auto psi_minus_interp_forward = makePsiMinusInterp(interpolation_matrix_trans< basis_params >);
template < BasisParams basis_params >
inline const auto psi_plus_der_back = makePsiPlusDer(derivative_matrix< basis_params >);
template < BasisParams basis_params >
inline const auto psi_minus_der_back = makePsiMinusDer(derivative_matrix< basis_params >);
template < BasisParams basis_params >
inline const auto psi_plus_der_forward = makePsiPlusDer(derivative_matrix_trans< basis_params >);
template < BasisParams basis_params >
inline const auto psi_minus_der_forward = makePsiMinusDer(derivative_matrix_trans< basis_params >);

template < int full_rows, int psi_p_rows, int psi_m_rows, typename EigenMap >
auto makeEO(EigenMap&& in)
{
    constexpr int in_cols = std::remove_cvref_t< EigenMap >::ColsAtCompileTime;
    using e_t             = Eigen::Matrix< val_t, psi_p_rows, in_cols >;
    using o_t             = Eigen::Matrix< val_t, psi_m_rows, in_cols >;
    auto retval           = std::pair< e_t, o_t >{};
    auto& [e, o]          = retval;
    for (int c = 0; c != in_cols; ++c)
    {
        for (int r = 0; r != psi_m_rows; ++r)
        {
            const int   ri = full_rows - r - 1;
            const val_t v1 = in(r, c);
            const val_t v2 = in(ri, c);
            e(r, c)        = .5 * (v1 + v2);
            o(r, c)        = .5 * (v1 - v2);
        }
        if constexpr (psi_m_rows < psi_p_rows)
            e(psi_m_rows, c) = in(psi_m_rows, c);
    }
    return retval;
}

template < typename EigenMap, int in_cols, int e_prime_cols, int o_prime_cols, typename Combine >
void reconstructOddEven(EigenMap&&                                           out,
                        const Eigen::Matrix< val_t, in_cols, e_prime_cols >& e_prime,
                        const Eigen::Matrix< val_t, in_cols, o_prime_cols >& o_prime,
                        Combine&&                                            combine)
{
    constexpr int full_cols = e_prime_cols + o_prime_cols;
    for (int c = 0; c != o_prime_cols; ++c)
    {
        const int ci = full_cols - c - 1;
        for (int r = 0; r != in_cols; ++r)
        {
            const val_t v1 = e_prime(r, c);
            const val_t v2 = o_prime(r, c);
            combine(out(r, c), v1 + v2);
            combine(out(r, ci), v1 - v2);
        }
    }
    if constexpr (e_prime_cols > o_prime_cols)
        combine(out.col(o_prime_cols), e_prime.template rightCols< 1 >());
}

template < int full_rows, typename EigenMapIn, typename EigenMapOut, typename PsiPlus, typename PsiMinus >
void sumFactSweepInterpOddEvenBlock(EigenMapIn&&    in,
                                    EigenMapOut&&   out,
                                    const PsiPlus&  psi_plus,
                                    const PsiMinus& psi_minus)
{
    constexpr int psi_p_rows = PsiPlus::RowsAtCompileTime;
    constexpr int psi_m_rows = PsiMinus::RowsAtCompileTime;
    const auto [e, o]        = makeEO< full_rows, psi_p_rows, psi_m_rows >(std::forward< EigenMapIn >(in));
    const auto e_prime       = (e.transpose() * psi_plus).eval();
    const auto o_prime       = (o.transpose() * psi_minus).eval();
    reconstructOddEven(std::forward< EigenMapOut >(out), e_prime, o_prime, [](auto&& a, auto&& b) { a = b; });
}

template < int full_rows,
           typename EigenMapIn,
           typename EigenMapOut,
           typename PsiPlus,
           typename PsiMinus,
           typename Combine >
void sumFactSweepDerOddEvenBlockImpl(
    EigenMapIn&& in, EigenMapOut&& out, const PsiPlus& psi_plus, const PsiMinus& psi_minus, Combine&& combine)
{
    constexpr int psi_p_rows = PsiPlus::RowsAtCompileTime;
    constexpr int psi_m_rows = PsiMinus::RowsAtCompileTime;
    const auto [e, o]        = makeEO< full_rows, psi_p_rows, psi_m_rows >(std::forward< EigenMapIn >(in));
    const auto e_prime       = (e.transpose() * psi_plus).eval();
    const auto o_prime       = (o.transpose() * psi_minus).eval();
    reconstructOddEven(std::forward< EigenMapOut >(out), o_prime, e_prime, std::forward< Combine >(combine));
}

template < int full_rows, typename EigenMapIn, typename EigenMapOut, typename PsiPlus, typename PsiMinus >
void sumFactSweepDerAssignOddEvenBlock(EigenMapIn&&    in,
                                       EigenMapOut&&   out,
                                       const PsiPlus&  psi_plus,
                                       const PsiMinus& psi_minus)
{
    sumFactSweepDerOddEvenBlockImpl< full_rows >(
        std::forward< EigenMapIn >(in), std::forward< EigenMapOut >(out), psi_plus, psi_minus, [](auto&& a, auto&& b) {
            a = b;
        });
}

template < int full_rows, typename EigenMapIn, typename EigenMapOut, typename PsiPlus, typename PsiMinus >
void sumFactSweepDerAccumulateOddEvenBlock(EigenMapIn&&    in,
                                           EigenMapOut&&   out,
                                           const PsiPlus&  psi_plus,
                                           const PsiMinus& psi_minus)
{
    sumFactSweepDerOddEvenBlockImpl< full_rows >(
        std::forward< EigenMapIn >(in), std::forward< EigenMapOut >(out), psi_plus, psi_minus, [](auto&& a, auto&& b) {
            a += b;
        });
}

template < BasisParams basis_params, typename EigenMapIn, typename EigenMapOut >
void sumFactSweepBackInterpOddEven(EigenMapIn&& in, EigenMapOut&& out)
{
    constexpr int in_cols        = std::remove_cvref_t< decltype(in) >::ColsAtCompileTime;
    constexpr int block_size     = util::simd_width / 2;
    constexpr int num_blocks     = in_cols / block_size;
    constexpr int remainder_size = in_cols % block_size;
    constexpr int psi_rows       = basis_params.n_bases1d();

    for (int block = 0; block != num_blocks; ++block)
        sumFactSweepInterpOddEvenBlock< psi_rows >(in.template middleCols< block_size >(block * block_size),
                                                   out.template middleRows< block_size >(block * block_size),
                                                   psi_plus_interp_back< basis_params >,
                                                   psi_minus_interp_back< basis_params >);
    if constexpr (remainder_size > 0)
        sumFactSweepInterpOddEvenBlock< psi_rows >(in.template rightCols< remainder_size >(),
                                                   out.template bottomRows< remainder_size >(),
                                                   psi_plus_interp_back< basis_params >,
                                                   psi_minus_interp_back< basis_params >);
}

template < BasisParams basis_params, typename EigenMapIn, typename EigenMapOut >
void sumFactSweepForwardInterpAssignOddEven(EigenMapIn&& in, EigenMapOut&& out)
{
    constexpr int in_cols        = std::remove_cvref_t< decltype(in) >::ColsAtCompileTime;
    constexpr int block_size     = util::simd_width / 2;
    constexpr int num_blocks     = in_cols / block_size;
    constexpr int remainder_size = in_cols % block_size;
    constexpr int psi_rows       = basis_params.n_qps1d();

    for (int block = 0; block != num_blocks; ++block)
        sumFactSweepInterpOddEvenBlock< psi_rows >(in.template middleCols< block_size >(block * block_size),
                                                   out.template middleRows< block_size >(block * block_size),
                                                   psi_plus_interp_forward< basis_params >,
                                                   psi_minus_interp_forward< basis_params >);
    if constexpr (remainder_size > 0)
        sumFactSweepInterpOddEvenBlock< psi_rows >(in.template rightCols< remainder_size >(),
                                                   out.template bottomRows< remainder_size >(),
                                                   psi_plus_interp_forward< basis_params >,
                                                   psi_minus_interp_forward< basis_params >);
}

template < BasisParams basis_params, typename EigenMapIn, typename EigenMapOut >
void sumFactSweepBackDerOddEven(EigenMapIn&& in, EigenMapOut&& out)
{
    constexpr int in_cols        = std::remove_cvref_t< decltype(in) >::ColsAtCompileTime;
    constexpr int block_size     = util::simd_width / 2;
    constexpr int num_blocks     = in_cols / block_size;
    constexpr int remainder_size = in_cols % block_size;
    constexpr int psi_rows       = basis_params.n_bases1d();

    for (int block = 0; block != num_blocks; ++block)
        sumFactSweepDerAssignOddEvenBlock< psi_rows >(in.template middleCols< block_size >(block * block_size),
                                                      out.template middleRows< block_size >(block * block_size),
                                                      psi_plus_der_back< basis_params >,
                                                      psi_minus_der_back< basis_params >);
    if constexpr (remainder_size > 0)
        sumFactSweepDerAssignOddEvenBlock< psi_rows >(in.template rightCols< remainder_size >(),
                                                      out.template bottomRows< remainder_size >(),
                                                      psi_plus_der_back< basis_params >,
                                                      psi_minus_der_back< basis_params >);
}

template < BasisParams basis_params, typename EigenMapIn, typename EigenMapOut >
void sumFactSweepForwardDerAccumulateOddEven(EigenMapIn&& in, EigenMapOut&& out)
{
    constexpr int in_cols        = std::remove_cvref_t< decltype(in) >::ColsAtCompileTime;
    constexpr int block_size     = util::simd_width / 2;
    constexpr int num_blocks     = in_cols / block_size;
    constexpr int remainder_size = in_cols % block_size;
    constexpr int psi_rows       = basis_params.n_qps1d();

    for (int block = 0; block != num_blocks; ++block)
        sumFactSweepDerAccumulateOddEvenBlock< psi_rows >(in.template middleCols< block_size >(block * block_size),
                                                          out.template middleRows< block_size >(block * block_size),
                                                          psi_plus_der_forward< basis_params >,
                                                          psi_minus_der_forward< basis_params >);
    if constexpr (remainder_size > 0)
        sumFactSweepDerAccumulateOddEvenBlock< psi_rows >(in.template rightCols< remainder_size >(),
                                                          out.template bottomRows< remainder_size >(),
                                                          psi_plus_der_forward< basis_params >,
                                                          psi_minus_der_forward< basis_params >);
}

template < BasisParams basis_params, size_t num_fields, size_t dim >
    requires(dim == 2 or dim == 3)
class SumFactBufferHelper
{
    static constexpr size_t n_bases  = util::integralPower(basis_params.n_bases1d(), dim);
    static constexpr size_t n_quads  = util::integralPower(basis_params.n_qps1d(), dim);
    static constexpr size_t max_size = std::max(n_bases, n_quads);

public:
    static constexpr size_t buf_size = max_size * num_fields;
    using buffer_t                   = std::array< val_t, buf_size >;
    using buf_array_t                = std::array< buffer_t, dim + 1 >;

private:
    template < typename Map >
    static auto makeBufferViewsImpl(buf_array_t& bufs)
    {
        constexpr auto make_map = [](buffer_t& buf) {
            return Map{buf.data()};
        };
        // Can't std::transform in a civilized way, since maps are not default-constructible
        if constexpr (dim == 2)
            return std::array{make_map(bufs[0]), make_map(bufs[1]), make_map(bufs[2])};
        else
            return std::array{make_map(bufs[0]), make_map(bufs[1]), make_map(bufs[2]), make_map(bufs[3])};
    }

public:
    static auto makeBufferViewsAtQuadsRowMajor(buf_array_t& bufs)
    {
        using map_t = Eigen::Map< util::eigen::RowMajorMatrix< val_t, n_quads, num_fields > >;
        return makeBufferViewsImpl< map_t >(bufs);
    }
    static auto makeBufferViewsAtQuadsColMajor(buf_array_t& bufs)
    {
        using map_t = Eigen::Map< Eigen::Matrix< val_t, n_quads, num_fields > >;
        return makeBufferViewsImpl< map_t >(bufs);
    }
};

template < KernelParams params, AssemblyOptions asm_opts, el_o_t EO >
inline constexpr auto make_basis_params = BasisParams{.basis_order = EO,
                                                      .quad_order  = 2 * asm_opts.order(EO),
                                                      .basis_type  = asm_opts.basis_type,
                                                      .quad_type   = asm_opts.quad_type};
template < KernelParams params, AssemblyOptions asm_opts, el_o_t EO >
inline constexpr auto make_geom_basis_params = BasisParams{.basis_order = 1,
                                                           .quad_order  = 2 * asm_opts.order(EO),
                                                           .basis_type  = basis::BasisType::Lagrange,
                                                           .quad_type   = asm_opts.quad_type};

template < BasisParams basis_params, size_t num_fields, std::invocable< std::span< val_t > > Fill >
auto sumFactBackQuad(Fill&& fill) -> SumFactBufferHelper< basis_params, num_fields, 2 >::buf_array_t
{
    L3STER_PROFILE_FUNCTION;
    constexpr int n_bases = basis_params.n_bases1d();
    constexpr int n_quads = basis_params.n_qps1d();

    // Eigen::Map types used to perform the individual sweeps
    using x0_t = Eigen::Map< Eigen::Matrix< val_t, n_bases, num_fields * n_bases > >;
    using x1_t = Eigen::Map< Eigen::Matrix< val_t, n_bases, num_fields * n_quads > >;
    using y0_t = Eigen::Map< Eigen::Matrix< val_t, num_fields * n_bases, n_quads > >;
    using y1_t = Eigen::Map< Eigen::Matrix< val_t, num_fields * n_quads, n_quads > >;

    // Requisite buffers
    typename SumFactBufferHelper< basis_params, num_fields, 2 >::buf_array_t retval;
    typename SumFactBufferHelper< basis_params, num_fields, 2 >::buffer_t    temp;
    auto& [r0, r1, r2] = retval;

    // Fill with the input for the backwards transformation
    std::invoke(std::forward< Fill >(fill), std::span< val_t >{r1});

    // Sum-factorization sweeps, note the buffer reuse pattern
    sumFactSweepBackInterp< basis_params >(x0_t{r1.data()}, y0_t{temp.data()});
    sumFactSweepBackDer< basis_params >(x1_t{temp.data()}, y1_t{r2.data()});
    sumFactSweepBackInterp< basis_params >(x1_t{temp.data()}, y1_t{r0.data()});
    sumFactSweepBackDer< basis_params >(x0_t{r1.data()}, y0_t{temp.data()});
    sumFactSweepBackInterp< basis_params >(x1_t{temp.data()}, y1_t{r1.data()});

    return retval;
}

template < BasisParams basis_params, size_t num_fields, std::invocable< std::span< val_t > > Fill >
auto sumFactBackHex(Fill&& fill) -> SumFactBufferHelper< basis_params, num_fields, 3 >::buf_array_t
{
    L3STER_PROFILE_FUNCTION;
    constexpr int n_bases = basis_params.n_bases1d();
    constexpr int n_quads = basis_params.n_qps1d();

    // Eigen::Map types used to perform the individual sweeps
    using x0_t = Eigen::Map< Eigen::Matrix< val_t, n_bases, num_fields * n_bases * n_bases > >;
    using x1_t = Eigen::Map< Eigen::Matrix< val_t, n_bases, num_fields * n_quads * n_bases > >;
    using x2_t = Eigen::Map< Eigen::Matrix< val_t, n_bases, num_fields * n_quads * n_quads > >;
    using y0_t = Eigen::Map< Eigen::Matrix< val_t, num_fields * n_bases * n_bases, n_quads > >;
    using y1_t = Eigen::Map< Eigen::Matrix< val_t, num_fields * n_quads * n_bases, n_quads > >;
    using y2_t = Eigen::Map< Eigen::Matrix< val_t, num_fields * n_quads * n_quads, n_quads > >;

    // Requisite buffers
    typename SumFactBufferHelper< basis_params, num_fields, 3 >::buf_array_t retval;
    typename SumFactBufferHelper< basis_params, num_fields, 3 >::buffer_t    temp;
    auto& [r0, r1, r2, r3] = retval;

    // Fill with the input for the backwards transformation
    std::invoke(std::forward< Fill >(fill), std::span< val_t >{r3});

    // Sum-factorization sweeps, note the buffer reuse pattern
    sumFactSweepBackInterp< basis_params >(x0_t{r3.data()}, y0_t{r0.data()});
    sumFactSweepBackDer< basis_params >(x0_t{r3.data()}, y0_t{r1.data()});
    sumFactSweepBackInterp< basis_params >(x1_t{r1.data()}, y1_t{r3.data()});
    sumFactSweepBackInterp< basis_params >(x2_t{r3.data()}, y2_t{r1.data()});
    sumFactSweepBackDer< basis_params >(x1_t{r0.data()}, y1_t{r3.data()});
    sumFactSweepBackInterp< basis_params >(x2_t{r3.data()}, y2_t{r2.data()});
    sumFactSweepBackInterp< basis_params >(x1_t{r0.data()}, y1_t{temp.data()});
    sumFactSweepBackInterp< basis_params >(x2_t{temp.data()}, y2_t{r0.data()});
    sumFactSweepBackDer< basis_params >(x2_t{temp.data()}, y2_t{r3.data()});

    return retval;
}

template < BasisParams basis_params, el_o_t EO >
auto computeGeomDataLin(const mesh::ElementData< mesh::ElementType::Quad, EO >& el_data)
{
    L3STER_PROFILE_FUNCTION;
    const auto fill_coords = [&](std::span< val_t > to_fill) {
        constexpr size_t num_verts = 4;
        for (auto&& [ind, vertex] : el_data.vertices | std::views::enumerate)
        {
            const auto i           = static_cast< size_t >(ind);
            to_fill[i]             = vertex.x();
            to_fill[i + num_verts] = vertex.y();
        }
    };
    return sumFactBackQuad< basis_params, 2 >(fill_coords);
}

template < BasisParams basis_params, el_o_t EO >
auto computeGeomDataLin(const mesh::ElementData< mesh::ElementType::Hex, EO >& el_data)
{
    L3STER_PROFILE_FUNCTION;
    const auto fill_coords = [&](std::span< val_t > to_fill) {
        constexpr size_t num_verts = 8;
        for (auto&& [ind, vertex] : el_data.vertices | std::views::enumerate)
        {
            const auto i               = static_cast< size_t >(ind);
            to_fill[i]                 = vertex.x();
            to_fill[i + num_verts]     = vertex.y();
            to_fill[i + 2 * num_verts] = vertex.z();
        }
    };
    return sumFactBackHex< basis_params, 3 >(fill_coords);
}

inline auto makeJacobianMat(val_t dx_dxi, val_t dx_deta, val_t dy_dxi, val_t dy_deta)
{
    auto retval  = Eigen::Matrix< val_t, 2, 2 >{};
    retval(0, 0) = dx_dxi;
    retval(0, 1) = dx_deta;
    retval(1, 0) = dy_dxi;
    retval(1, 1) = dy_deta;
    return retval;
}

inline auto makeJacobianMat(val_t dx_dxi,
                            val_t dx_deta,
                            val_t dx_dzeta,
                            val_t dy_dxi,
                            val_t dy_deta,
                            val_t dy_dzeta,
                            val_t dz_dxi,
                            val_t dz_deta,
                            val_t dz_dzeta)
{
    auto retval  = Eigen::Matrix< val_t, 3, 3 >{};
    retval(0, 0) = dx_dxi;
    retval(0, 1) = dx_deta;
    retval(0, 2) = dx_dzeta;
    retval(1, 0) = dy_dxi;
    retval(1, 1) = dy_deta;
    retval(1, 2) = dy_dzeta;
    retval(2, 0) = dz_dxi;
    retval(2, 1) = dz_deta;
    retval(2, 2) = dz_dzeta;
    return retval;
}

template < KernelParams params, typename Field >
auto evalFieldVals(const Field& v, int qi)
{
    auto retval = typename KernelInterface< params >::FieldVals{};
    for (size_t i = 0; i != params.n_fields; ++i)
        retval[i] = v(qi, i);
    return retval;
}

template < KernelParams params, typename Field >
auto evalFieldDers(const Field& dxi, const Field& deta, const Eigen::Matrix< val_t, 2, 2 >& jac_inv, int qi)
{
    auto retval    = typename KernelInterface< params >::FieldDers{};
    auto& [dx, dy] = retval;
    for (size_t i = 0; i != params.n_fields; ++i)
    {
        const auto der_xi  = dxi(qi, i);
        const auto der_eta = deta(qi, i);
        dx[i]              = jac_inv(0, 0) * der_xi + jac_inv(1, 0) * der_eta;
        dy[i]              = jac_inv(0, 1) * der_xi + jac_inv(1, 1) * der_eta;
    }
    return retval;
}

template < KernelParams params, typename Field >
auto evalFieldDers(
    const Field& dxi, const Field& deta, const Field& dzeta, const Eigen::Matrix< val_t, 3, 3 >& jac_inv, int qi)
{
    auto retval        = typename KernelInterface< params >::FieldDers{};
    auto& [dx, dy, dz] = retval;
    for (size_t i = 0; i != params.n_fields; ++i)
    {
        const auto der_xi   = dxi(qi, i);
        const auto der_eta  = deta(qi, i);
        const auto der_zeta = dzeta(qi, i);
        dx[i]               = jac_inv(0, 0) * der_xi + jac_inv(1, 0) * der_eta + jac_inv(2, 0) * der_zeta;
        dy[i]               = jac_inv(0, 1) * der_xi + jac_inv(1, 1) * der_eta + jac_inv(2, 1) * der_zeta;
        dz[i]               = jac_inv(0, 2) * der_xi + jac_inv(1, 2) * der_eta + jac_inv(2, 2) * der_zeta;
    }
    return retval;
}

template < typename Kernel, KernelParams params, el_o_t EO, AssemblyOptions asm_opts >
auto evalAtQuadQPs(typename SumFactBufferHelper< make_basis_params< params, asm_opts, EO >,
                                                 params.n_unknowns * params.n_rhs + params.n_fields,
                                                 2 >::buf_array_t&         back_trans_result,
                   const DomainEquationKernel< Kernel, params >&           kernel,
                   const mesh::ElementData< mesh::ElementType::Quad, EO >& element_data,
                   val_t                                                   time,
                   util::ConstexprValue< asm_opts >)
{
    L3STER_PROFILE_FUNCTION;
    constexpr auto basis_params     = make_basis_params< params, asm_opts, EO >;
    constexpr auto num_operands     = params.n_unknowns * params.n_rhs;
    constexpr int  num_fields_total = num_operands + params.n_fields;
    using basis_helper_t            = SumFactBufferHelper< basis_params, num_fields_total, 2 >;
    auto [vals, dv_dxi, dv_deta]    = basis_helper_t::makeBufferViewsAtQuadsRowMajor(back_trans_result);

    constexpr auto geom_basis_params = make_geom_basis_params< params, asm_opts, EO >;
    using geom_helper_t              = SumFactBufferHelper< geom_basis_params, 2, 2 >;
    auto geom_trans_result           = detail::computeGeomDataLin< geom_basis_params >(element_data);
    const auto [x, dx_dxi, dx_deta]  = geom_helper_t::makeBufferViewsAtQuadsRowMajor(geom_trans_result);

    using return_helper_t = SumFactBufferHelper< basis_params, num_operands, 2 >;
    typename return_helper_t::buf_array_t retval;
    auto [r0, r1, r2] = return_helper_t::makeBufferViewsAtQuadsColMajor(retval);

    using operand_map_t    = Eigen::Map< Eigen::Matrix< val_t, params.n_unknowns, params.n_rhs > >;
    const auto&    weights = quad::getReferenceQuadrature< basis_params.quad_type, basis_params.quad_order >().weights;
    constexpr auto n_qps1d = basis_params.n_qps1d();
    for (int qy = 0; qy != n_qps1d; ++qy)
    {
        const auto wy = weights[qy];
        for (int qx = 0; qx != n_qps1d; ++qx)
        {
            const auto qi           = qy * n_qps1d + qx;
            const auto jacobian_mat = makeJacobianMat(dx_dxi(qi, 0), dx_deta(qi, 0), dx_dxi(qi, 1), dx_deta(qi, 1));
            const auto jac_inv      = jacobian_mat.inverse().eval();
            const auto kernel_input = typename KernelInterface< params >::DomainInput{
                .field_vals = evalFieldVals< params >(vals.template rightCols< params.n_fields >(), qi),
                .field_ders = evalFieldDers< params >(dv_dxi.template rightCols< params.n_fields >(),
                                                      dv_deta.template rightCols< params.n_fields >(),
                                                      jac_inv,
                                                      qi),
                .point      = SpaceTimePoint{Point< 3 >{x(qi, 0), x(qi, 1), 0.}, time}};
            const auto kernel_result = std::invoke(kernel, kernel_input);
            const auto& [A0, A1, A2] = kernel_result.operators;
            const auto D1            = (A1 * jac_inv(0, 0) + A2 * jac_inv(0, 1)).eval();
            const auto D2            = (A1 * jac_inv(1, 0) + A2 * jac_inv(1, 1)).eval();
            const auto t0            = operand_map_t{vals.row(qi).data()};
            const auto t1            = operand_map_t{dv_dxi.row(qi).data()};
            const auto t2            = operand_map_t{dv_deta.row(qi).data()};
            const auto jacobian      = jacobian_mat.determinant();
            const auto wx            = weights[qx];
            const auto wgt           = wx * wy * jacobian;
            const auto t             = (wgt * (A0 * t0 + D1 * t1 + D2 * t2)).eval();
            r0.row(qi)               = (A0.transpose() * t).reshaped().transpose();
            r1.row(qi)               = (D1.transpose() * t).reshaped().transpose();
            r2.row(qi)               = (D2.transpose() * t).reshaped().transpose();
        }
    }

    return retval;
}

template < typename Kernel, KernelParams params, el_o_t EO, AssemblyOptions asm_opts >
auto evalAtHexQPs(typename SumFactBufferHelper< make_basis_params< params, asm_opts, EO >,
                                                params.n_unknowns * params.n_rhs + params.n_fields,
                                                3 >::buf_array_t&        back_trans_result,
                  const DomainEquationKernel< Kernel, params >&          kernel,
                  const mesh::ElementData< mesh::ElementType::Hex, EO >& element_data,
                  val_t                                                  time,
                  util::ConstexprValue< asm_opts >)
{
    L3STER_PROFILE_FUNCTION;
    constexpr auto basis_params            = make_basis_params< params, asm_opts, EO >;
    constexpr auto num_operands            = params.n_unknowns * params.n_rhs;
    constexpr int  num_fields_total        = num_operands + params.n_fields;
    using basis_helper_t                   = SumFactBufferHelper< basis_params, num_fields_total, 3 >;
    auto [vals, dv_dxi, dv_deta, dv_dzeta] = basis_helper_t::makeBufferViewsAtQuadsRowMajor(back_trans_result);

    constexpr auto geom_basis_params          = make_geom_basis_params< params, asm_opts, EO >;
    using geom_helper_t                       = SumFactBufferHelper< geom_basis_params, 3, 3 >;
    auto geom_trans_result                    = detail::computeGeomDataLin< geom_basis_params >(element_data);
    const auto [x, dx_dxi, dx_deta, dx_dzeta] = geom_helper_t::makeBufferViewsAtQuadsRowMajor(geom_trans_result);

    using return_helper_t = SumFactBufferHelper< basis_params, num_operands, 3 >;
    typename return_helper_t::buf_array_t retval;
    auto [r0, r1, r2, r3] = return_helper_t::makeBufferViewsAtQuadsColMajor(retval);

    using operand_map_t    = Eigen::Map< Eigen::Matrix< val_t, params.n_unknowns, params.n_rhs > >;
    const auto&    weights = quad::getReferenceQuadrature< basis_params.quad_type, basis_params.quad_order >().weights;
    constexpr auto n_qps1d = basis_params.n_qps1d();
    for (int qz = 0; qz != n_qps1d; ++qz)
    {
        const auto wz = weights[qz];
        for (int qy = 0; qy != n_qps1d; ++qy)
        {
            const auto wy = weights[qy];
            for (int qx = 0; qx != n_qps1d; ++qx)
            {
                const auto qi           = qz * n_qps1d * n_qps1d + qy * n_qps1d + qx;
                const auto jacobian_mat = makeJacobianMat(dx_dxi(qi, 0),
                                                          dx_deta(qi, 0),
                                                          dx_dzeta(qi, 0),
                                                          dx_dxi(qi, 1),
                                                          dx_deta(qi, 1),
                                                          dx_dzeta(qi, 1),
                                                          dx_dxi(qi, 2),
                                                          dx_deta(qi, 2),
                                                          dx_dzeta(qi, 2));
                const auto jac_inv      = jacobian_mat.inverse().eval();
                const auto kernel_input = typename KernelInterface< params >::DomainInput{
                    .field_vals = evalFieldVals< params >(vals.template rightCols< params.n_fields >(), qi),
                    .field_ders = evalFieldDers< params >(dv_dxi.template rightCols< params.n_fields >(),
                                                          dv_deta.template rightCols< params.n_fields >(),
                                                          dv_dzeta.template rightCols< params.n_fields >(),
                                                          jac_inv,
                                                          qi),
                    .point      = SpaceTimePoint{Point< 3 >{x(qi, 0), x(qi, 1), x(qi, 2)}, time}};
                const auto kernel_result     = std::invoke(kernel, kernel_input);
                const auto& [A0, A1, A2, A3] = kernel_result.operators;
                const auto D1                = (A1 * jac_inv(0, 0) + A2 * jac_inv(0, 1) + A3 * jac_inv(0, 2)).eval();
                const auto D2                = (A1 * jac_inv(1, 0) + A2 * jac_inv(1, 1) + A3 * jac_inv(1, 2)).eval();
                const auto D3                = (A1 * jac_inv(2, 0) + A2 * jac_inv(2, 1) + A3 * jac_inv(2, 2)).eval();
                const auto t0                = operand_map_t{vals.row(qi).data()};
                const auto t1                = operand_map_t{dv_dxi.row(qi).data()};
                const auto t2                = operand_map_t{dv_deta.row(qi).data()};
                const auto t3                = operand_map_t{dv_dzeta.row(qi).data()};
                const auto jacobian          = jacobian_mat.determinant();
                const auto wx                = weights[qx];
                const auto wgt               = wx * wy * wz * jacobian;
                const auto t                 = (wgt * (A0 * t0 + D1 * t1 + D2 * t2 + D3 * t3)).eval();
                r0.row(qi)                   = (A0.transpose() * t).reshaped().transpose();
                r1.row(qi)                   = (D1.transpose() * t).reshaped().transpose();
                r2.row(qi)                   = (D2.transpose() * t).reshaped().transpose();
                r3.row(qi)                   = (D3.transpose() * t).reshaped().transpose();
            }
        }
    }

    return retval;
}

template < BasisParams basis_params, size_t num_fields, size_t N >
void sumFactForwardQuad(typename SumFactBufferHelper< basis_params, num_fields, 2 >::buf_array_t& ts,
                        std::array< val_t, N >&                                                   temp)
{
    L3STER_PROFILE_FUNCTION;
    constexpr int n_bases = basis_params.n_bases1d();
    constexpr int n_quads = basis_params.n_qps1d();

    static_assert(N >= num_fields * n_quads * n_quads);
    static_assert(N >= num_fields * n_bases * n_bases);

    // Eigen::Map types used to perform the individual sweeps
    using x0_t = Eigen::Map< Eigen::Matrix< val_t, n_quads, num_fields * n_quads > >;
    using x1_t = Eigen::Map< Eigen::Matrix< val_t, n_quads, num_fields * n_bases > >;
    using y0_t = Eigen::Map< Eigen::Matrix< val_t, num_fields * n_quads, n_bases > >;
    using y1_t = Eigen::Map< Eigen::Matrix< val_t, num_fields * n_bases, n_bases > >;

    // Sum-factorization sweeps, note the result is written to t0
    auto& [t0, t1, t2] = ts;
    sumFactSweepForwardInterpAssign< basis_params >(x0_t{t0.data()}, y0_t{temp.data()});
    sumFactSweepForwardDerAccumulate< basis_params >(x0_t{t1.data()}, y0_t{temp.data()});
    sumFactSweepForwardInterpAssign< basis_params >(x1_t{temp.data()}, y1_t{t0.data()});
    sumFactSweepForwardInterpAssign< basis_params >(x0_t{t2.data()}, y0_t{temp.data()});
    sumFactSweepForwardDerAccumulate< basis_params >(x1_t{temp.data()}, y1_t{t0.data()});
}

template < BasisParams basis_params, size_t num_fields, size_t N >
void sumFactForwardHex(typename SumFactBufferHelper< basis_params, num_fields, 3 >::buf_array_t& ts,
                       std::array< val_t, N >&                                                   temp)
{
    L3STER_PROFILE_FUNCTION;
    constexpr int n_bases = basis_params.n_bases1d();
    constexpr int n_quads = basis_params.n_qps1d();

    static_assert(N >= num_fields * n_quads * n_quads * n_quads);
    static_assert(N >= num_fields * n_bases * n_bases * n_bases);

    // Eigen::Map types used to perform the individual sweeps
    using x0_t = Eigen::Map< Eigen::Matrix< val_t, n_quads, num_fields * n_quads * n_quads > >;
    using x1_t = Eigen::Map< Eigen::Matrix< val_t, n_quads, num_fields * n_bases * n_quads > >;
    using x2_t = Eigen::Map< Eigen::Matrix< val_t, n_quads, num_fields * n_bases * n_bases > >;
    using y0_t = Eigen::Map< Eigen::Matrix< val_t, num_fields * n_quads * n_quads, n_bases > >;
    using y1_t = Eigen::Map< Eigen::Matrix< val_t, num_fields * n_bases * n_quads, n_bases > >;
    using y2_t = Eigen::Map< Eigen::Matrix< val_t, num_fields * n_bases * n_bases, n_bases > >;

    // Sum-factorization sweeps, note the result is written to t0
    auto& [t0, t1, t2, t3] = ts;
    sumFactSweepForwardInterpAssign< basis_params >(x0_t{t0.data()}, y0_t{temp.data()});
    sumFactSweepForwardDerAccumulate< basis_params >(x0_t{t1.data()}, y0_t{temp.data()});
    sumFactSweepForwardInterpAssign< basis_params >(x1_t{temp.data()}, y1_t{t1.data()});
    sumFactSweepForwardInterpAssign< basis_params >(x0_t{t2.data()}, y0_t{temp.data()});
    sumFactSweepForwardDerAccumulate< basis_params >(x1_t{temp.data()}, y1_t{t1.data()});
    sumFactSweepForwardInterpAssign< basis_params >(x2_t{t1.data()}, y2_t{t0.data()});
    sumFactSweepForwardInterpAssign< basis_params >(x0_t{t3.data()}, y0_t{t1.data()});
    sumFactSweepForwardInterpAssign< basis_params >(x1_t{t1.data()}, y1_t{t3.data()});
    sumFactSweepForwardDerAccumulate< basis_params >(x2_t{t3.data()}, y2_t{t0.data()});
}

template < typename Kernel,
           KernelParams                         params,
           el_o_t                               EO,
           AssemblyOptions                      asm_opts,
           std::invocable< std::span< val_t > > Fill,
           std::invocable< std::span< val_t > > YScatter >
void sumFactImpl(const DomainEquationKernel< Kernel, params >&           kernel,
                 const mesh::ElementData< mesh::ElementType::Quad, EO >& element_data,
                 Fill&&                                                  fill,
                 YScatter&&                                              y_scatter,
                 val_t                                                   time,
                 util::ConstexprValue< asm_opts >                        asm_opts_ctwrpr)
{
    constexpr auto basis_params     = make_basis_params< params, asm_opts, EO >;
    constexpr auto num_operands     = params.n_unknowns * params.n_rhs;
    constexpr int  num_fields_total = num_operands + params.n_fields;

    auto back_bufs    = sumFactBackQuad< basis_params, num_fields_total >(std::forward< Fill >(fill));
    auto forward_bufs = evalAtQuadQPs(back_bufs, kernel, element_data, time, asm_opts_ctwrpr);
    sumFactForwardQuad< basis_params, num_operands >(forward_bufs, back_bufs.front()); // use second argument as temp
    util::requestStackSize< 2 * (sizeof back_bufs + sizeof forward_bufs) >();          // this works backwards in time

    std::invoke(y_scatter, std::span< val_t >{forward_bufs.front()});
}

template < typename Kernel,
           KernelParams                         params,
           el_o_t                               EO,
           AssemblyOptions                      asm_opts,
           std::invocable< std::span< val_t > > Fill,
           std::invocable< std::span< val_t > > YScatter >
void sumFactImpl(const DomainEquationKernel< Kernel, params >&          kernel,
                 const mesh::ElementData< mesh::ElementType::Hex, EO >& element_data,
                 Fill&&                                                 fill,
                 YScatter&&                                             y_scatter,
                 val_t                                                  time,
                 util::ConstexprValue< asm_opts >                       asm_opts_ctwrpr)
{
    constexpr auto basis_params     = make_basis_params< params, asm_opts, EO >;
    constexpr auto num_operands     = params.n_unknowns * params.n_rhs;
    constexpr int  num_fields_total = num_operands + params.n_fields;

    auto back_bufs    = sumFactBackHex< basis_params, num_fields_total >(std::forward< Fill >(fill));
    auto forward_bufs = evalAtHexQPs(back_bufs, kernel, element_data, time, asm_opts_ctwrpr);
    sumFactForwardHex< basis_params, num_operands >(forward_bufs, back_bufs.front()); // use second argument as temp
    util::requestStackSize< 2 * (sizeof back_bufs + sizeof forward_bufs) >();         // this works backwards in time

    std::invoke(y_scatter, std::span< val_t >{forward_bufs.front()});
}
} // namespace detail

/// Evaluate the action of a domain operator using the sum-factorization technique
/// \param kernel Kernel defining the operator
/// \param element Local view of quad element for which the operator is evaluated
/// \param x_gather Callable used to populate the vector on which the operator is to be evaluated. Takes a span where
/// the values are to be written, and a stride defining the spacing between entries corresponding to a given node
/// \param field_access FieldAccess object used to read the external fields needed by the kernel
/// \param y_scatter Callable used to scatter the result into the destination (multi)vector. Takes a span containing the
/// evaluation result, and a stride defining the spacing between entries corresponding to a given node
/// \param time Time instance passed to the kernel
template < typename Kernel,
           KernelParams                         params,
           mesh::ElementType                    ET,
           el_o_t                               EO,
           AssemblyOptions                      asm_opts = {},
           std::invocable< std::span< val_t > > XGather,
           std::invocable< std::span< val_t > > YScatter >
void evalLocalOperatorSumFact(const DomainEquationKernel< Kernel, params >& kernel,
                              const mesh::LocalElementView< ET, EO >&       element,
                              XGather&&                                     x_gather,
                              const post::FieldAccess< params.n_fields >&   field_access,
                              YScatter&&                                    y_scatter,
                              util::ConstexprValue< asm_opts >              asm_opts_ctwrpr = {},
                              val_t                                         time            = 0.)
{
    L3STER_PROFILE_FUNCTION;
    const auto fill = [&](std::span< val_t > to_fill) {
        constexpr int num_fields_total = params.n_unknowns * params.n_rhs + params.n_fields;
        constexpr int num_nodes        = mesh::Element< ET, EO >::n_nodes;
        using field_map_t              = Eigen::Map< Eigen::Matrix< val_t, num_nodes, num_fields_total > >;
        std::invoke(std::forward< XGather >(x_gather), to_fill);
        if constexpr (params.n_fields > 0)
            field_access.fill(field_map_t{to_fill.data()}.template rightCols< params.n_fields >(),
                              element.getLocalNodes());
    };
    detail::sumFactImpl(kernel, element.getData(), fill, std::forward< YScatter >(y_scatter), time, asm_opts_ctwrpr);
}
} // namespace lstr::algsys
#endif // L3STER_ALGSYS_SUMFACTORIZATION_HPP
