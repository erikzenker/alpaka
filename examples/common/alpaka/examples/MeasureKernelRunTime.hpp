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

#include <alpaka/alpaka.hpp>

#include <utility>          // std::forward

namespace alpaka
{
    namespace examples
    {
        //-----------------------------------------------------------------------------
        //! \return The run time of the given kernel.
        //-----------------------------------------------------------------------------
        template<
            typename TExec,
            typename TKernelFunctor,
            typename... TArgs>
        auto measureKernelRunTimeMs(
            TExec const & exec,
            TKernelFunctor && kernelFunctor,
            TArgs && ... args)
        -> std::chrono::milliseconds::rep
        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
            std::cout
                << "measureKernelRunTime("
                << " exec: " << typeid(TExec).name()
                << ")" << std::endl;
#endif
            // Take the time prior to the execution.
            auto const tpStart(std::chrono::high_resolution_clock::now());

            // Execute the kernel functor.
            exec(
                std::forward<TKernelFunctor>(kernelFunctor),
                std::forward<TArgs>(args)...);

            // Wait for the stream to finish the kernel execution to measure its run time.
            alpaka::wait::wait(alpaka::stream::getStream(exec));

            // Take the time after the execution.
            auto const tpEnd(std::chrono::high_resolution_clock::now());

            auto const durElapsed(tpEnd - tpStart);

            // Return the duration.
            return std::chrono::duration_cast<std::chrono::milliseconds>(durElapsed).count();
        }
    }
}
