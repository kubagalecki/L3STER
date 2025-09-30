#include "Common.hpp"
#include "l3ster/comm/ImportExport.hpp"
#include "l3ster/util/Functional.hpp"
#include "l3ster/util/ScopeGuards.hpp"

using namespace lstr;
using namespace lstr::comm;

namespace sr = std::ranges;
namespace sv = std::views;

using GID                       = long;
using Scalar                    = GID;
constexpr size_t mv_cols        = 5;
constexpr size_t elems_per_rank = 1000;

void runTest(const MpiComm&                    comm,
             Import< Scalar >&                 importer,
             Export< Scalar >&                 exporter,
             const util::ArrayOwner< Scalar >& x,
             util::ArrayOwner< Scalar >&       y)
{
    const auto comm_size = static_cast< size_t >(comm.getSize());
    const auto rank      = static_cast< size_t >(comm.getRank());
    const bool is_last   = rank == comm_size - 1;

    auto x_cols = x | sv::chunk(x.size() / mv_cols);
    auto y_cols = y | sv::chunk(y.size() / mv_cols);

    const auto eval_elem = [&](ptrdiff_t el) {
        for (auto&& [xc, yc] : sv::zip(x_cols, y_cols))
        {
            std::atomic_ref y1{yc[el]}, y2{yc[el + 1]};
            const auto      update = -xc[el] + xc[el + 1];
            y1.fetch_add(update, std::memory_order_relaxed);
            y2.fetch_add(-update, std::memory_order_relaxed);
        }
    };

    sr::fill(y, 0);
    exporter.postRecvs(comm);
    importer.postComms(comm);
    if (is_last)
    {
        exporter.postSends(comm);
        util::tbb::parallelFor(sv::iota(0u, elems_per_rank), eval_elem);
        exporter.wait(util::AtomicSumInto{});
        importer.wait();
    }
    else
    {
        util::tbb::parallelFor(sv::iota(0u, elems_per_rank - 1), [&](ptrdiff_t el) {
            eval_elem(el);
            if (importer.tryReceive())
            {
                eval_elem(elems_per_rank - 1);
                exporter.postSends(comm);
            }
        });
        if (not importer.testReceive())
        {
            importer.waitReceive();
            eval_elem(elems_per_rank - 1);
            exporter.postSends(comm);
        }
        exporter.wait(util::AtomicSumInto{});
        importer.wait();
    }
}

int main(int argc, char* argv[])
{
    const auto          max_par_guard = util::MaxParallelismGuard{4};
    util::MpiScopeGuard mpi_scope_guard{argc, argv};
    MpiComm             comm{MPI_COMM_WORLD};

    const auto comm_size = static_cast< size_t >(comm.getSize());
    const auto rank      = static_cast< size_t >(comm.getRank());
    const bool is_first  = rank == 0;
    const bool is_last   = rank == comm_size - 1;

    const auto first_ind  = static_cast< GID >(elems_per_rank * rank);
    auto       owned_inds = util::ArrayOwner< GID >(elems_per_rank + is_last);
    std::iota(owned_inds.begin(), owned_inds.end(), first_ind);
    auto shared_inds = util::ArrayOwner< GID >(not is_last);
    if (not shared_inds.empty())
        shared_inds.front() = first_ind + static_cast< GID >(elems_per_rank);

    const auto                 all_inds_sz  = owned_inds.size() + shared_inds.size();
    const auto                 total_vec_sz = mv_cols * all_inds_sz;
    util::ArrayOwner< Scalar > x(total_vec_sz), y(total_vec_sz);
    for (const auto& [i, x_col] : x | sv::chunk(all_inds_sz) | sv::enumerate)
    {
        auto is = sv::repeat(i + 1, owned_inds.size()) | sv::common;
        std::inclusive_scan(is.begin(), is.end(), x_col.begin(), std::plus{}, (i + 1) * elems_per_rank * rank);
        if (not is_last)
            x_col.back() = -42; // This needs to be imported
    }

    const auto ownership = util::SegmentedOwnership{first_ind, owned_inds.size(), shared_inds};
    auto       context   = std::make_shared< const ImportExportContext >(comm, ownership);
    auto       importer  = Import< Scalar >{context, 2 * mv_cols};
    auto       exporter  = Export< Scalar >{std::move(context), 2 * mv_cols};
    REQUIRE(importer.getNumVecs() == 2 * mv_cols);
    REQUIRE(exporter.getNumVecs() == 2 * mv_cols);
    importer.setNumVecs(mv_cols);
    exporter.setNumVecs(mv_cols);
    REQUIRE(importer.getNumVecs() == mv_cols);
    REQUIRE(exporter.getNumVecs() == mv_cols);
    CHECK_THROWS(importer.setNumVecs(3 * mv_cols));
    CHECK_THROWS(exporter.setNumVecs(3 * mv_cols));
    importer.setOwned(x, all_inds_sz);
    importer.setShared(std::span{x}.subspan(owned_inds.size()), all_inds_sz);
    exporter.setOwned(y, all_inds_sz);
    exporter.setShared(std::span{y}.subspan(owned_inds.size()), all_inds_sz);

    const auto check_results = [&] {
        for (const auto& [i, y_vec_view] : y | sv::chunk(all_inds_sz) | sv::enumerate)
        {
            if (is_first)
                REQUIRE(y_vec_view.front() == i + 1);
            const size_t internal_start = is_first, internal_finish = elems_per_rank;
            for (size_t index = internal_start; index != internal_finish; ++index)
                REQUIRE(y_vec_view[index] == 0);
            if (is_last)
                REQUIRE(y_vec_view.back() == -(i + 1));
        }
    };

    constexpr int reps = 20; // Repeat the test this many times to try to catch multithreading issues
    for (int rep = 0; rep != reps; ++rep)
    {
        runTest(comm, importer, exporter, x, y);
        check_results();
    }
}