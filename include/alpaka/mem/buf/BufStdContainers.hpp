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

#include <alpaka/core/Common.hpp>       // ALPAKA_FN_HOST

// FIXME: This include can lead to circular include problems!
#include <alpaka/dev/DevCpu.hpp>        // dev::DevCpu

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused
#include <boost/predef.h>               // workarounds

#include <type_traits>                  // std::enable_if, std::is_array, std::extent
#include <vector>                       // std::vector
#include <array>                        // std::array

namespace alpaka
{
    //-----------------------------------------------------------------------------
    // Trait specializations for fixed size arrays.
    //
    // This allows the usage of multidimensional compile time arrays e.g. int[4][3] as argument to memory ops.
    //-----------------------------------------------------------------------------
    /*namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The fixed size array device type trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct DevType<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
            {
                using type = dev::DevCpu;
            };

            //#############################################################################
            //! The fixed size array device get trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct GetDev<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    TFixedSizeArray const & buf)
                -> dev::DevCpu
                {
                    // \FIXME: CUDA device?
                    return dev::cpu::getDev();
                }
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The fixed size array dimension getter trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct DimType<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
            {
                using type = dim::DimInt<std::rank<TFixedSizeArray>::value>;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The fixed size array width get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TFixedSizeArray>
            struct GetExtent<
                TIdx,
                TFixedSizeArray,
                typename std::enable_if<
                    std::is_array<TFixedSizeArray>::value
                    && (std::rank<TFixedSizeArray>::value > TIdx::value)
                    && (std::extent<TFixedSizeArray, TIdx::value>::value > 0u)>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static constexpr auto getExtent(
                    TFixedSizeArray const & //extents
                )
                -> size::Size<TFixedSizeArray>
                {
                    //boost::ignore_unused(extents);
                    return std::extent<TFixedSizeArray, TIdx::value>::value;
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
                //! The fixed size array memory element type get trait specialization.
                //#############################################################################
                template<
                    typename TFixedSizeArray>
                struct ElemType<
                    TFixedSizeArray,
                    typename std::enable_if<
                        std::is_array<TFixedSizeArray>::value>::type>
                {
                    using type = typename std::remove_all_extents<TFixedSizeArray>::type;
                };

                //#############################################################################
                //! The fixed size array buf trait specialization.
                //#############################################################################
                template<
                    typename TFixedSizeArray>
                struct GetBuf<
                    TFixedSizeArray,
                    typename std::enable_if<
                        std::is_array<TFixedSizeArray>::value>::type>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getBuf(
                        TFixedSizeArray const & buf)
                    -> TFixedSizeArray const &
                    {
                        return buf;
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getBuf(
                        TFixedSizeArray & buf)
                    -> TFixedSizeArray &
                    {
                        return buf;
                    }
                };

                //#############################################################################
                //! The fixed size array native pointer get trait specialization.
                //#############################################################################
                template<
                    typename TFixedSizeArray>
                struct GetPtrNative<
                    TFixedSizeArray,
                    typename std::enable_if<
                        std::is_array<TFixedSizeArray>::value>::type>
                {
                    using TElem = typename std::remove_all_extents<TFixedSizeArray>::type;

                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto getPtrNative(
                        TFixedSizeArray const & buf)
                    -> TElem const *
                    {
                        return buf;
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static auto getPtrNative(
                        TFixedSizeArray & buf)
                    -> TElem *
                    {
                        return buf;
                    }
                };

                //#############################################################################
                //! The fixed size array pitch get trait specialization.
                //#############################################################################
                template<
                    typename TFixedSizeArray>
                struct GetPitchBytes<
                    std::integral_constant<std::size_t, std::rank<TFixedSizeArray>::value - 1u>,
                    TFixedSizeArray,
                    typename std::enable_if<
                        std::is_array<TFixedSizeArray>::value
                        && (std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 1u>::value > 0u)>::type>
                {
                    using TElem = typename std::remove_all_extents<TFixedSizeArray>::type;

                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_HOST_ACC static constexpr auto getPitchBytes(
                        TFixedSizeArray const &)
                    -> size::Size<TFixedSizeArray>
                    {
                        return sizeof(TElem) * std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 1u>::value;
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
            //! The fixed size array offset get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TFixedSizeArray>
            struct GetOffset<
                TIdx,
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getOffset(
                    TFixedSizeArray const &)
                -> size::Size<TFixedSizeArray>
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
            //! The std::vector size type trait specialization.
            //#############################################################################
            template<
                typename TFixedSizeArray>
            struct SizeType<
                TFixedSizeArray,
                typename std::enable_if<std::is_array<TFixedSizeArray>::value>::type>
            {
                using type = std::size_t;
            };
        }
    }*/

    //-----------------------------------------------------------------------------
    // Trait specializations for std::array.
    //-----------------------------------------------------------------------------
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The std::array device type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                std::size_t TuiSize>
            struct DevType<
                std::array<TElem, TuiSize>>
            {
                using type = dev::DevCpu;
            };

            //#############################################################################
            //! The std::array device get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                std::size_t TuiSize>
            struct GetDev<
                std::array<TElem, TuiSize>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    std::array<TElem, TuiSize> const & buf)
                -> dev::DevCpu
                {
                    return dev::cpu::getDev();
                }
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The std::array dimension getter trait specialization.
            //#############################################################################
            template<
                typename TElem,
                std::size_t TuiSize>
            struct DimType<
                std::array<TElem, TuiSize>>
            {
                using type = dim::DimInt<1u>;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The std::array width get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                std::size_t TuiSize>
            struct GetExtent<
                std::integral_constant<std::size_t, 0u>,
                std::array<TElem, TuiSize>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static constexpr auto getExtent(
                    std::array<TElem, TuiSize> const & /*extents*/)
                -> size::Size<std::array<TElem, TuiSize>>
                {
                    // C++14
                    /*boost::ignore_unused(extents);*/
                    return TuiSize;
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
                //! The std::array memory element type get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    std::size_t TuiSize>
                struct ElemType<
                    std::array<TElem, TuiSize>>
                {
                    using type = TElem;
                };

                //#############################################################################
                //! The std::array buf trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    std::size_t TuiSize>
                struct GetBuf<
                    std::array<TElem, TuiSize>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getBuf(
                        std::array<TElem, TuiSize> const & buf)
                    -> std::array<TElem, TuiSize> const &
                    {
                        return buf;
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getBuf(
                        std::array<TElem, TuiSize> & buf)
                    -> std::array<TElem, TuiSize> &
                    {
                        return buf;
                    }
                };

                //#############################################################################
                //! The std::array native pointer get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    std::size_t TuiSize>
                struct GetPtrNative<
                    std::array<TElem, TuiSize>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        std::array<TElem, TuiSize> const & buf)
                    -> TElem const *
                    {
                        return buf.data();
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        std::array<TElem, TuiSize> & buf)
                    -> TElem *
                    {
                        return buf.data();
                    }
                };

                //#############################################################################
                //! The std::array pitch get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    std::size_t TuiSize>
                struct GetPitchBytes<
                    std::integral_constant<std::size_t, 0u>,
                    std::array<TElem, TuiSize>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPitchBytes(
                        std::array<TElem, TuiSize> const & pitch)
                    -> size::Size<std::array<TElem, TuiSize>>
                    {
                        return sizeof(TElem) * pitch.size();
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
            //! The std::array offset get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TElem,
                std::size_t TuiSize>
            struct GetOffset<
                TIdx,
                std::array<TElem, TuiSize>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getOffset(
                    std::array<TElem, TuiSize> const &)
                -> size::Size<std::array<TElem, TuiSize>>
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
            //! The std::vector size type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                std::size_t TuiSize>
            struct SizeType<
                std::array<TElem, TuiSize>>
            {
                using type = std::size_t;
            };
        }
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for std::vector.
    //-----------------------------------------------------------------------------
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The std::vector device type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TAllocator>
            struct DevType<
                std::vector<TElem, TAllocator>>
            {
                using type = dev::DevCpu;
            };

            //#############################################################################
            //! The std::vector device get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TAllocator>
            struct GetDev<
                std::vector<TElem, TAllocator>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    std::vector<TElem, TAllocator> const & buf)
                -> dev::DevCpu
                {
                    return dev::cpu::getDev();
                }
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The std::vector dimension getter trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TAllocator>
            struct DimType<
                std::vector<TElem, TAllocator>>
            {
                using type = dim::DimInt<1u>;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The std::vector width get trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TAllocator>
            struct GetExtent<
                std::integral_constant<std::size_t, 0u>,
                std::vector<TElem, TAllocator>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getExtent(
                    std::vector<TElem, TAllocator> const & extents)
                -> size::Size<std::vector<TElem, TAllocator>>
                {
                    return extents.size();
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
                //! The std::vector memory element type get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TAllocator>
                struct ElemType<
                    std::vector<TElem, TAllocator>>
                {
                    using type = TElem;
                };

                //#############################################################################
                //! The std::vector buf trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TAllocator>
                struct GetBuf<
                    std::vector<TElem, TAllocator>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getBuf(
                        std::vector<TElem, TAllocator> const & buf)
                    -> std::vector<TElem, TAllocator> const &
                    {
                        return buf;
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getBuf(
                        std::vector<TElem, TAllocator> & buf)
                    -> std::vector<TElem, TAllocator> &
                    {
                        return buf;
                    }
                };

                //#############################################################################
                //! The std::vector native pointer get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TAllocator>
                struct GetPtrNative<
                    std::vector<TElem, TAllocator>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        std::vector<TElem, TAllocator> const & buf)
                    -> TElem const *
                    {
                        return buf.data();
                    }
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        std::vector<TElem, TAllocator> & buf)
                    -> TElem *
                    {
                        return buf.data();
                    }
                };

                //#############################################################################
                //! The std::vector pitch get trait specialization.
                //#############################################################################
                template<
                    typename TElem,
                    typename TAllocator>
                struct GetPitchBytes<
                    std::integral_constant<std::size_t, 0u>,
                    std::vector<TElem, TAllocator>>
                {
                    //-----------------------------------------------------------------------------
                    //!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPitchBytes(
                        std::vector<TElem, TAllocator> const & pitch)
                    -> size::Size<std::vector<TElem, TAllocator>>
                    {
                        return sizeof(TElem) * pitch.size();
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
            //! The std::vector offset get trait specialization.
            //#############################################################################
            template<
                typename TIdx,
                typename TElem,
                typename TAllocator>
            struct GetOffset<
                TIdx,
                std::vector<TElem, TAllocator>>
            {
                //-----------------------------------------------------------------------------
                //!
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getOffset(
                    std::vector<TElem, TAllocator> const &)
                -> size::Size<std::vector<TElem, TAllocator>>
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
            //! The std::vector size type trait specialization.
            //#############################################################################
            template<
                typename TElem,
                typename TAllocator>
            struct SizeType<
                std::vector<TElem, TAllocator>>
            {
                using type = std::size_t;
            };
        }
    }
}
