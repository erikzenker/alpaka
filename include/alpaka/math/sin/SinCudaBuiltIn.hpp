/**
* \file
* Copyright 2014-2015 Benjasin Worpitz
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

#include <alpaka/math/sin/Traits.hpp>   // Sin

//#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

#include <type_traits>                  // std::enable_if, std::is_floating_point
#include <math_functions.hpp>           // ::sin

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library sin.
        //#############################################################################
        class SinCudaBuiltIn
        {
        public:
            using SinBase = SinCudaBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library sin trait specialization.
            //#############################################################################
            template<
                typename TArg>
            struct Sin<
                SinCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                ALPAKA_FN_ACC_NO_CUDA static auto sin(
                    SinCudaBuiltIn const & sin,
                    TArg const & arg)
                -> decltype(::sin(arg))
                {
                    //boost::ignore_unused(sin);
                    return ::sin(arg);
                }
            };
        }
    }
}
