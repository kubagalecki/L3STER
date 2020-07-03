// This header includes the entire L3STER library

// First, include dependencies
#include "../eigen/Eigen/Dense"

// Meta-programming utility functions
#include "../src/utility/Meta.hpp"

// Type definitions for entire library
#include "../src/definitions/Typedefs.h"
#include "../src/definitions/Constants.hpp"
#include "../src/mesh/ElementTypes.hpp"
#include "../src/quadrature/QuadratureTypes.h"
#include "../src/definitions/Aliases.hpp"

// Library-wide utilities
#include "../src/utility/Polynomial.hpp"

// Node is independent of the Element hierarchy
#include "../src/mesh/Node.hpp"

// Element hierarchy
#include "../src/mesh/ElementTraits.hpp"
#include "../src/mesh/Element.hpp"
#include "../src/mesh/Domain.hpp"
#include "../src/mesh/MeshPartition.hpp"

// Quadratures, dependent on elements
#include "../src/quadrature/QuadratureTraits.hpp"
#include "../src/quadrature/Quadrature.hpp"
#include "../src/quadrature/QuadratureGenerator.hpp"

// Mesh, dependent on elements and nodes
#include "../src/mesh/Mesh.hpp"
#include "../src/mesh/ReadMesh.hpp"
