#ifndef L3STER_HPP
#define L3STER_HPP

// This header includes the entire L3STER library

// First, include dependencies
#include "../eigen/Eigen/Dense"

// Meta-programming utility functions
#include "../src/utility/Meta.hpp"

// Type definitions for entire library
#include "../src/definitions/Typedefs.h"
#include "../src/definitions/Constants.hpp"
#include "../src/mesh/ElementTypes.hpp"
#include "../src/definitions/Aliases.hpp"
#include "../src/quadrature/QuadratureTypes.h"

// Library-wide utilities
#include "../src/utility/Polynomial.hpp"

// Node is independent of the Element hierarchy
#include "../src/mesh/Node.hpp"

// Element hierarchy
#include "../src/mesh/ElementTraits.hpp"
#include "../src/mesh/Element.hpp"
#include "../src/mesh/Domain.hpp"
#include "../src/mesh/MeshPartition.hpp"
#include "../src/mesh/Mesh.hpp"
#include "../src/mesh/ReadMesh.hpp"
#include "../src/mesh/BoundaryElementView.hpp"
#include "../src/mesh/BoundaryView.hpp"

// Quadratures, dependent on elements
#include "../src/quadrature/Quadrature.hpp"
#include "../src/quadrature/ReferenceQuadratureTraits.hpp"
#include "../src/quadrature/ReferenceQuadrature.hpp"
#include "../src/quadrature/QuadratureGenerator.hpp"

#endif // L3STER_HPP
