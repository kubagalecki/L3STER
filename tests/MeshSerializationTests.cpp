#include "comm/SerializeMesh.hpp"
#include "mesh/ReadMesh.hpp"
#include "mesh/primitives/CubeMesh.hpp"

#include "TestDataPath.h"
#include "catch2/catch.hpp"

TEST_CASE("Mesh serialization", "[mesh-serial]")
{
    const auto  mesh            = lstr::readMesh(L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square.msh), lstr::gmsh_tag);
    const auto& part            = mesh.getPartitions()[0];
    const auto  serialized_mesh = lstr::SerializedPartition{part};
}
