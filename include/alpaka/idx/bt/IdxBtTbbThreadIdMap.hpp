/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

//#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

#include <alpaka/idx/Traits.hpp>            // idx::getIdx

#include <boost/core/ignore_unused.hpp>     // boost::ignore_unused

#include <thread>                           // std::thread
#include <map>                              // std::map

#include <tbb/tbb_thread.h>                 // tbb::thread::id

namespace alpaka
{
    namespace idx
    {
        namespace bt
        {
            //#############################################################################
            //! The threads accelerator index provider.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            class IdxBtTbbThreadIdMap
            {
            public:
                using IdxBtBase = IdxBtTbbThreadIdMap;

                using ThreadIdToIdxMap = std::map<tbb::tbb_thread::id, Vec<TDim, TSize>>;

                //-----------------------------------------------------------------------------
                //! Constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA IdxBtTbbThreadIdMap(
                    ThreadIdToIdxMap const & mThreadToIndices) :
                    m_threadToIndexMap(mThreadToIndices)
                {}
                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA IdxBtTbbThreadIdMap(IdxBtTbbThreadIdMap const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA IdxBtTbbThreadIdMap(IdxBtTbbThreadIdMap &&) = delete;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(IdxBtTbbThreadIdMap const &) -> IdxBtTbbThreadIdMap & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA auto operator=(IdxBtTbbThreadIdMap &&) -> IdxBtTbbThreadIdMap & = delete;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~IdxBtTbbThreadIdMap() = default;

            public:
                ThreadIdToIdxMap const & m_threadToIndexMap;   //!< The mapping of thread id's to thread indices.
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU threads accelerator index dimension get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct DimType<
                idx::bt::IdxBtTbbThreadIdMap<TDim, TSize>>
            {
                using type = TDim;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU threads accelerator block thread index get trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct GetIdx<
                idx::bt::IdxBtTbbThreadIdMap<TDim, TSize>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                //-----------------------------------------------------------------------------
                template<
                    typename TWorkDiv>
                ALPAKA_FN_ACC_NO_CUDA static auto getIdx(
                    idx::bt::IdxBtTbbThreadIdMap<TDim, TSize> const & idx,
                    TWorkDiv const & workDiv)
                -> Vec<TDim, TSize>
                {
                    boost::ignore_unused(workDiv);
                    auto const threadId(tbb::this_tbb_thread::get_id());
                    auto const threadEntry(idx.m_threadToIndexMap.find(threadId));
                    assert(threadEntry != idx.m_threadToIndexMap.end());
                    return threadEntry->second;
                }
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU threads accelerator block thread index size type trait specialization.
            //#############################################################################
            template<
                typename TDim,
                typename TSize>
            struct SizeType<
                idx::bt::IdxBtTbbThreadIdMap<TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}

//#endif
