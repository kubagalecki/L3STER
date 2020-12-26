// Utility for handling polynomials: computing roots, etc.
/*
 * In the following, polynomials are assumed to be in the form of:
 * p(x) = sum[i = 0 : N] ( ai * x^i ), where N is the order of the polynomial
 */

#ifndef L3STER_MATH_POLYNOMIAL_HPP
#define L3STER_MATH_POLYNOMIAL_HPP

#include "Eigen/Dense"

#include "defs/Typedefs.h"

#include <algorithm>
#include <array>
#include <exception>
#include <initializer_list>
#include <iterator>
#include <numeric>
#include <string>
#include <type_traits>

namespace lstr::math
{
//////////////////////////////////////////////////////////////////////////////////////////////
//                                      POLYNOMIAL CLASS                                    //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
Class containing the polynomial coefficients
*/
template < types::poly_o_t N >
class Polynomial final
{
public:
    template < types::poly_o_t I >
    friend class Polynomial;

    // Aliases
    using array_t = std::array< types::val_t, N + 1 >;

    // Ctors & Dtors
    Polynomial(std::initializer_list< types::val_t >);

    template < typename... Types >
    Polynomial(const Types&... args) : coefs(args...)
    {}

    // Access
    types::val_t* data()
    {
        return coefs.data(); // access the underlying array
    }

    // Polynomial manipulations
    // evaluate polynomial
    types::val_t eval(const types::val_t&);

    Polynomial< N - 1 > polyder();

    Polynomial< N + 1 > polyint();

    std::array< std::complex< types::val_t >, N > roots();

    // Operators
    Polynomial< N > operator+=(types::val_t _a)
    {
        coefs.front() += _a; // increment by cosntant
        return *this;
    }

    Polynomial< N > operator*=(types::val_t);

private:
    array_t coefs;
};

// Forward declare specialization of template
template <>
class Polynomial< 0 >;

// Initializer list constructor
template < types::poly_o_t N >
Polynomial< N >::Polynomial(std::initializer_list< types::val_t > init_list)
{
    if (init_list.size() != N + 1)
    {
        throw(std::length_error(
            "Incorrect number of coefficients for polynomial of specified order. Should be: " +
            std::to_string(N + 1) + ", is: " + std::to_string(init_list.size()) + '\n'));
    }

    std::move(init_list.begin(), init_list.end(), coefs.begin());
}

// Polynomial evaluation
template < types::poly_o_t N >
types::val_t Polynomial< N >::eval(const types::val_t& x)
{
    // evaluate p(x)
    auto ret_val     = coefs.front();
    auto current_exp = x;

    auto op = [&](types::val_t c) -> void {
        ret_val += current_exp * c;
        current_exp *= x;
    };

    std::for_each(++coefs.begin(), --coefs.end(), op);

    ret_val += current_exp * coefs.back();
    return ret_val;
}

// Polynomial dervative
template < types::poly_o_t N >
Polynomial< N - 1 > Polynomial< N >::polyder()
{
    // Handle conversion to proxy type Polynomial<0>
    if (N == 1)
        return Polynomial< 0 >{coefs[1]};

    auto ret_val = Polynomial< N - 1 >{};

    auto current_exp = N;

    auto op = [&](const typename array_t::value_type& a_old) {
        return a_old * (current_exp--);
    };

    std::transform(coefs.crbegin(), std::prev(coefs.crend()), ret_val.coefs.rbegin(), op);

    return ret_val;
}

// Polynomial integral, C = 0
template < types::poly_o_t N >
Polynomial< N + 1 > Polynomial< N >::polyint()
{
    auto ret_val          = Polynomial< N + 1 >{};
    ret_val.coefs.front() = 0.;
    auto current_exp      = 1;

    auto op = [&](const typename array_t::value_type& a_old) {
        return a_old / (current_exp++);
    };

    std::transform(coefs.begin(), coefs.end(), std::next(ret_val.coefs.begin()), op);
    return ret_val;
}

// Polynomial roots (complex valued)
template < types::poly_o_t N >
std::array< std::complex< types::val_t >, N > Polynomial< N >::roots()
{
    using complex_t = std::complex< types::val_t >;

    // If linear function, x0 = -b/a
    if (N == 1)
        return std::array< complex_t, N >{complex_t{coefs.front() / coefs.back(), 0.}};

    // Create and populate companion matrix
    auto comp_mat = Eigen::Matrix< types::val_t, N, N >{};

    for (auto i = 0; i < N; i++)
    {
        for (auto j = 0; j < N; j++)
        {
            if (i == j + 1)
            {
                comp_mat(i, j) = 1.;
            }
            else
            {
                if (j == N - 1)
                {
                    comp_mat(i, j) = -coefs[i] / coefs.back();
                }
                else
                {
                    comp_mat(i, j) = 0;
                }
            }
        }
    }

    // Get Eigen::Vector of eigenvalues
    auto eig = comp_mat.eigenvalues();

    // Copy to the returned array
    auto ret_val = std::array< complex_t, N >{};
    std::copy(eig.begin(), eig.end(), ret_val.begin());

    // Sort by absolute value
    std::sort(ret_val.begin(), ret_val.end(), [](complex_t a, complex_t b) {
        return std::abs(a) < std::abs(b);
    });

    return ret_val;
}

// Scaling
template < types::poly_o_t N >
Polynomial< N > Polynomial< N >::operator*=(types::val_t _a)
{
    auto op = [&](types::val_t c) {
        c *= _a;
    };
    std::for_each(coefs.begin(), coefs.end(), op);
    return *this;
}

// Specialization for order 0 polynomial
template <>
class Polynomial< 0 >
{
public:
    // Ctors & Dtors
    Polynomial() : a0(0) {}

    Polynomial(const types::val_t& _a0) : a0(_a0) {}

    // Access
    types::val_t* data()
    {
        return &a0; // access the underlying coefficient
    }

    // Polynomial manipulations
    types::val_t eval(const types::val_t&) { return a0; }

    Polynomial< 0 > polyder() { return Polynomial< 0 >{0.}; }

    Polynomial< 1 > polyint() { return Polynomial< 1 >{0., a0}; }

    // Operators
    Polynomial< 0 > operator+=(types::val_t _a)
    {
        a0 += _a;
        return *this;
    }

    Polynomial< 0 > operator*=(types::val_t _a)
    {
        a0 *= _a;
        return *this;
    }

private:
    types::val_t a0;

public:
    // Although not used, empty coefs array is required for correct conversion
    // This is a bit of a dirty hack
    std::array< types::val_t, 0 > coefs = {};
};

//////////////////////////////////////////////////////////////////////////////////////////////
//                                          FUNCTIONS                                       //
//////////////////////////////////////////////////////////////////////////////////////////////
// Lagrange interpolation
/*
Returns the polynomial p of order N, such that for all 0 <= i < N p(x[i]) = y[i]
*/
template < typename T, types::poly_o_t N = std::tuple_size< T >::value - 1 >
Polynomial< N > lagrangeFit(const T& x, const T& y)
{
    auto A = Eigen::Matrix< types::val_t, N + 1, N + 1 >{};
    auto b = Eigen::Matrix< types::val_t, N + 1, 1 >{};

    for (size_t i = 0; i <= N; i++)
    {
        b[i]    = y[i];
        A(i, 0) = 1.;

        for (size_t j = 1; j <= N; j++)
            A(i, j) = A(i, j - 1) * x[i];
    }

    Eigen::Matrix< types::val_t, N + 1, 1 > ret_coefs = A.colPivHouseholderQr().solve(b);

    auto ret_val = Polynomial< N >{};
    std::copy(ret_coefs.cbegin(), ret_coefs.cend(), ret_val.data());
    return ret_val;
}
} // namespace lstr::math

#endif // L3STER_MATH_POLYNOMIAL_HPP
