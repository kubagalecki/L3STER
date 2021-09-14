#ifndef L3STER_HPP
#define L3STER_HPP

// This header includes the entire L3STER library

#include "comm/DeserializeMesh.hpp"
#include "mesh/ConvertMeshToOrder.hpp"
#include "mesh/PartitionMesh.hpp"
#include "mesh/ReadMesh.hpp"
#include "quad/InvokeQuadrature.hpp"
#include "quad/QuadratureGenerator.hpp"

#include "alloc/NodeGlobalMemoryResource.hpp"
#include "comm/MpiComm.hpp"
#include "util/GlobalResource.hpp"
#include "util/HwlocWrapper.hpp"

#include "mesh/primitives/CubeMesh.hpp"

#include "mapping/ComputeBasisDerivative.hpp"

#endif // L3STER_HPP
