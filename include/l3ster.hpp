#ifndef L3STER_HPP
#define L3STER_HPP

// This header includes the entire L3STER library

#include "Eigen/Dense"

// Meta-programming utility functions
#include "utility/Meta.hpp"

// Type definitions for entire library
#include "definitions/Typedefs.h"
#include "definitions/Constants.hpp"
#include "mesh/ElementTypes.hpp"
#include "definitions/Aliases.hpp"
#include "quadrature/QuadratureTypes.h"

// Library-wide utilities
#include "utility/Polynomial.hpp"

// Node is independent of the Element hierarchy
#include "mesh/Node.hpp"

// Element hierarchy
#include "mesh/ElementTraits.hpp"
#include "mesh/Element.hpp"
#include "mesh/Domain.hpp"
#include "mesh/MeshPartition.hpp"
#include "mesh/Mesh.hpp"
#include "mesh/ReadMesh.hpp"
#include "mesh/BoundaryElementView.hpp"
#include "mesh/BoundaryView.hpp"

// Quadratures, dependent on elements
#include "quadrature/Quadrature.hpp"
#include "quadrature/ReferenceQuadratureTraits.hpp"
#include "quadrature/ReferenceQuadrature.hpp"
#include "quadrature/QuadratureGenerator.hpp"

#endif // L3STER_HPP
