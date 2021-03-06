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

#include <alpaka/core/Debug.hpp>

#include <type_traits>                      // std::enable_if

#if !defined(__CUDA_ARCH__)
    #include <boost/core/ignore_unused.hpp> // boost::ignore_unused
#endif
#include <boost/predef.h>                   // workarounds

//-----------------------------------------------------------------------------
//! Disable nvcc warning:
//! 'calling a __host__ function from __host__ __device__ function.'
//!
//! Usage:
//! ALPAKA_NO_HOST_ACC_WARNING
//! __device__ __host__ function_declaration()
//!
//! It is not possible to disable the warning for a __host__ function if there are calls of virtual functions inside.
//! For this case use a wrapper function.
//! WARNING: only use this method if there is no other way to create runnable code.
//! Most cases can solved by #ifdef __CUDA_ARCH__ or #ifdef __CUDACC__.
//-----------------------------------------------------------------------------
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
    #if BOOST_COMP_MSVC
        #define ALPAKA_NO_HOST_ACC_WARNING\
            __pragma(hd_warning_disable)
    #else
        #define ALPAKA_NO_HOST_ACC_WARNING\
            _Pragma("hd_warning_disable")
    #endif
#else
    #define ALPAKA_NO_HOST_ACC_WARNING
#endif

//-----------------------------------------------------------------------------
//! All functions that can be used on an accelerator have to be attributed with ALPAKA_FN_ACC_CUDA_ONLY or ALPAKA_FN_ACC.
//!
//! Usage:
//! ALPAKA_FN_ACC int add(int a, int b);
//-----------------------------------------------------------------------------
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDACC__)
    #define ALPAKA_FN_ACC_CUDA_ONLY __device__ __forceinline__
    #define ALPAKA_FN_ACC_NO_CUDA __host__ __forceinline__
    #define ALPAKA_FN_ACC __device__ __host__ __forceinline__
    #define ALPAKA_FN_HOST_ACC __device__ __host__ __forceinline__
    #define ALPAKA_FN_HOST __host__ __forceinline__
#else
    //#define ALPAKA_FN_ACC_CUDA_ONLY inline
    #define ALPAKA_FN_ACC_NO_CUDA inline
    #define ALPAKA_FN_ACC inline
    #define ALPAKA_FN_HOST_ACC inline
    #define ALPAKA_FN_HOST inline
#endif

//-----------------------------------------------------------------------------
//! Suggests unrolling of the directly following loop to the compiler.
//!
//! Usage:
//!  `ALPAKA_UNROLL
//!  for(...){...}`
// \TODO: Implement for other compilers.
//-----------------------------------------------------------------------------
#ifdef __CUDA_ARCH__
    #if BOOST_COMP_MSVC
        #define ALPAKA_UNROLL(...) __pragma(unroll __VA_ARGS__)
    #else
        #define ALPAKA_UNROLL_STRINGIFY(x) #x
        #define ALPAKA_UNROLL(...)  _Pragma(ALPAKA_UNROLL_STRINGIFY(unroll __VA_ARGS__))
    #endif
#else
    #if BOOST_COMP_INTEL || BOOST_COMP_IBM || BOOST_COMP_SUNPRO || BOOST_COMP_HPACC
        #define ALPAKA_UNROLL_STRINGIFY(x) #x
        #define ALPAKA_UNROLL(...)  _Pragma(ALPAKA_UNROLL_STRINGIFY(unroll(__VA_ARGS__)))
    #elif BOOST_COMP_PGI
        #define ALPAKA_UNROLL(...)  _Pragma("unroll")
    #else
        #define ALPAKA_UNROLL(...)
    #endif
#endif

//-----------------------------------------------------------------------------
//! Suggests vectorization of the directly following loop to the compiler.
//!
//! Usage:
//!  `ALPAKA_VECTORIZE
//!  for(...){...}`
// \TODO: Implement for other compilers.
// See: http://stackoverflow.com/questions/2706286/pragmas-swp-ivdep-prefetch-support-in-various-compilers
//-----------------------------------------------------------------------------
#if BOOST_COMP_INTEL || BOOST_COMP_HPACC
    #define ALPAKA_VECTORIZE(...)  _Pragma("ivdep")
#elif BOOST_COMP_PGI
    #define ALPAKA_VECTORIZE(...)  _Pragma("vector")
#elif BOOST_COMP_MSVC
    #define ALPAKA_VECTORIZE(...)  __pragma(loop(ivdep))
#else
    #define ALPAKA_VECTORIZE(...)
#endif

namespace alpaka
{
    namespace core
    {
        //#############################################################################
        //! A false_type being dependent on a ignored template parameter.
        //! This allows to use static_assert in uninstantiated template specializations without triggering.
        //#############################################################################
        template<
            typename T>
        struct DependentFalseType :
            std::false_type
        {};

        namespace detail
        {
            //#############################################################################
            //!
            //#############################################################################
            template<
                typename TArg,
                typename TSfinae = void>
            struct AssertValueUnsigned;
            //#############################################################################
            //!
            //#############################################################################
            template<
                typename TArg>
            struct AssertValueUnsigned<
                TArg,
                typename std::enable_if<!std::is_unsigned<TArg>::value>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto assertValueUnsigned(
                    TArg const & arg)
                -> void
                {
                    assert(arg >= 0);
                }
            };
            //#############################################################################
            //!
            //#############################################################################
            template<
                typename TArg>
            struct AssertValueUnsigned<
                TArg,
                typename std::enable_if<std::is_unsigned<TArg>::value>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto assertValueUnsigned(
                    TArg const & arg)
                -> void
                {
#if !defined(__CUDA_ARCH__)
                    boost::ignore_unused(arg);
#endif
                    // Nothing to do for unsigned types.
                }
            };
        }
        //-----------------------------------------------------------------------------
        //! This method checks integral values if they are greater or equal zero.
        //! The implementation prevents warnings for checking this for unsigned types.
        //-----------------------------------------------------------------------------
        template<
            typename TArg>
        ALPAKA_NO_HOST_ACC_WARNING
        ALPAKA_FN_HOST_ACC auto assertValueUnsigned(
            TArg const & arg)
        -> void
        {
            detail::AssertValueUnsigned<
                TArg>
            ::assertValueUnsigned(
                arg);
        }
    }
}
