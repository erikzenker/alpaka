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

#include <alpaka/mem/buf/Traits.hpp>    // dev::traits::DevType, DimType, GetExtent,Copy, GetOffset, ...

#include <alpaka/core/Vec.hpp>          // Vec<N>

namespace alpaka
{
    namespace mem
    {
        namespace buf
        {
            //#############################################################################
            //! The memory buffer wrapper used to wrap plain pointers.
            //#############################################################################
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            class BufPlainPtrWrapper final
            {
            public:
                using Dev = TDev;
                using Elem = TElem;
                using Dim = TDim;

            public:
                //-----------------------------------------------------------------------------
                //! Constructor
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TExtents>
                ALPAKA_FN_HOST_ACC BufPlainPtrWrapper(
                    TElem * pMem,
                    TDev const & dev,
                    TExtents const & extents = TExtents()) :
                        m_pMem(pMem),
                        m_Dev(dev),
                        m_vExtentsElements(extent::getExtentsVecEnd<TDim>(extents)),
                        m_uiPitchBytes(extent::getWidth(extents) * sizeof(TElem))
                {}

                //-----------------------------------------------------------------------------
                //! Constructor
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TExtents>
                ALPAKA_FN_HOST_ACC BufPlainPtrWrapper(
                    TElem * pMem,
                    TDev const dev,
                    TExtents const & extents,
                    TSize const & uiPitch) :
                        m_pMem(pMem),
                        m_Dev(dev),
                        m_vExtentsElements(extent::getExtentsVecEnd<TDim>(extents)),
                        m_uiPitchBytes(uiPitch)
                {}

                //-----------------------------------------------------------------------------
                //! Copy constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC BufPlainPtrWrapper(BufPlainPtrWrapper const &) = delete;
                //-----------------------------------------------------------------------------
                //! Move constructor.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC BufPlainPtrWrapper(BufPlainPtrWrapper &&) = default;
                //-----------------------------------------------------------------------------
                //! Copy assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC auto operator=(BufPlainPtrWrapper const &) -> BufPlainPtrWrapper & = delete;
                //-----------------------------------------------------------------------------
                //! Move assignment operator.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC auto operator=(BufPlainPtrWrapper &&) -> BufPlainPtrWrapper & = default;
                //-----------------------------------------------------------------------------
                //! Destructor.
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC ~BufPlainPtrWrapper() = default;

            public:
                TElem * m_pMem;
                TDev m_Dev;
                Vec<TDim, TSize> m_vExtentsElements;
                TSize m_uiPitchBytes;
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for BufPlainPtrWrapper.
    //-----------------------------------------------------------------------------
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The BufPlainPtrWrapper device type trait specialization.
            //#############################################################################
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            struct DevType<
                mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize>>
            {
                using type = TDev;
            };

            //#############################################################################
            //! The BufPlainPtrWrapper device get trait specialization.
            //#############################################################################
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            struct GetDev<
                mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getDev(
                    mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize> const & buf)
                    -> TDev
                {
                    return buf.m_Dev;
                }
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The BufPlainPtrWrapper dimension getter trait.
            //#############################################################################
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            struct DimType<
                mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize>>
            {
                using type = TDim;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The BufPlainPtrWrapper width get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            struct GetExtent<
                TIdx,
                mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize>,
                typename std::enable_if<(TDim::value > TIdx::value)>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getExtent(
                    mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize> const & extents)
                -> TSize
                {
                    return extents.m_vExtentsElements[TIdx::value];
                }
            };
        }
    }
    namespace mem
    {
        namespace view
        {
            namespace traits
            {
                //#############################################################################
                //! The BufPlainPtrWrapper memory element type get trait specialization.
                //#############################################################################
                template<
                    typename TDev,
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct ElemType<
                    buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize>>
                {
                    using type = TElem;
                };

                //#############################################################################
                //! The BufPlainPtrWrapper buf trait specialization.
                //#############################################################################
                template<
                    typename TDev,
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct GetBuf<
                    buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto getBuf(
                        buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize> const & buf)
                    -> buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize> const &
                    {
                        return buf;
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto getBuf(
                        buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize> & buf)
                    -> buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize> &
                    {
                        return buf;
                    }
                };

                //#############################################################################
                //! The BufPlainPtrWrapper native pointer get trait specialization.
                //#############################################################################
                template<
                    typename TDev,
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct GetPtrNative<
                    buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize>>
                {
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto getPtrNative(
                        buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize> const & buf)
                    -> TElem const *
                    {
                        return buf.m_pMem;
                    }
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto getPtrNative(
                        buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize> & buf)
                    -> TElem *
                    {
                        return buf.m_pMem;
                    }
                };

                //#############################################################################
                //! The BufPlainPtrWrapper memory pitch get trait specialization.
                //#############################################################################
                template<
                    typename TDev,
                    typename TElem,
                    typename TDim,
                    typename TSize>
                struct GetPitchBytes<
                    std::integral_constant<std::size_t, 0u>,
                    buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize>>
                {
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto getPitchBytes(
                        buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize> const & buf)
                    -> TSize
                    {
                        return buf.m_uiPitchBytes;
                    }
                };
            }
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The BufPlainPtrWrapper offset get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            struct GetOffset<
                TIdx,
                mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getOffset(
                    mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize> const &)
                -> TSize
                {
                    return 0u;
                }
            };
        }
    }
    namespace size
    {
        namespace traits
        {
            //#############################################################################
            //! The BufPlainPtrWrapper size type trait specialization.
            //#############################################################################
            template<
                typename TDev,
                typename TElem,
                typename TDim,
                typename TSize>
            struct SizeType<
                mem::buf::BufPlainPtrWrapper<TDev, TElem, TDim, TSize>>
            {
                using type = TSize;
            };
        }
    }
}