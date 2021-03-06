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

#include <alpaka/dev/DevCpu.hpp>            // dev::DevCpu
#include <alpaka/dev/Traits.hpp>            // GetDev
#include <alpaka/event/Traits.hpp>          // StreamEnqueue, ...
#include <alpaka/wait/Traits.hpp>           // CurrentThreadWaitFor
#include <alpaka/dev/Traits.hpp>            // GetDev

#include <boost/uuid/uuid.hpp>              // boost::uuids::uuid
#include <boost/uuid/uuid_generators.hpp>   // boost::uuids::random_generator

#include <mutex>                            // std::mutex
#include <condition_variable>               // std::condition_variable

namespace alpaka
{
    namespace event
    {
        namespace cpu
        {
            namespace detail
            {
                //#############################################################################
                //! The CPU device event implementation.
                //#############################################################################
                class EventCpuAsyncImpl final
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST EventCpuAsyncImpl(
                        dev::DevCpu const & dev) :
                            m_Uuid(boost::uuids::random_generator()()),
                            m_Dev(dev),
                            m_Mutex(),
                            m_bIsReady(true),
                            m_bIsWaitedFor(false),
                            m_uiNumCanceledEnqueues(0)
                    {}
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST EventCpuAsyncImpl(EventCpuAsyncImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST EventCpuAsyncImpl(EventCpuAsyncImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto operator=(EventCpuAsyncImpl const &) -> EventCpuAsyncImpl & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto operator=(EventCpuAsyncImpl &&) -> EventCpuAsyncImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST ~EventCpuAsyncImpl() noexcept
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
                    {
                        // If a event is enqueued to a stream and gets waited on but destructed before it is completed it is kept alive until completed.
                        // This can never happen.
                        assert(!m_bIsWaitedFor);
                    }
#else
                    = default;
#endif
                public:
                    boost::uuids::uuid const m_Uuid;                        //!< The unique ID.
                    dev::DevCpu const m_Dev;                                //!< The device this event is bound to.

                    std::mutex mutable m_Mutex;                             //!< The mutex used to synchronize access to the event.

                    bool m_bIsReady;                                        //!< If the event is not waiting within a stream (not enqueued or already completed).
                    std::condition_variable mutable m_ConditionVariable;    //!< The condition signaling the event completion.

                    bool m_bIsWaitedFor;                                    //!< If a (one or multiple) streams wait for this event. The event can not be changed (deleted/re-enqueued) until completion.

                    std::size_t m_uiNumCanceledEnqueues;                    //!< The number of successive re-enqueues while it was already in the queue. Reset on completion.
                };
            }
        }

        //#############################################################################
        //! The CPU device event.
        //#############################################################################
        class EventCpuAsync final
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST EventCpuAsync(
                dev::DevCpu const & dev) :
                    m_spEventCpuAsyncImpl(std::make_shared<cpu::detail::EventCpuAsyncImpl>(dev))
            {}
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST EventCpuAsync(EventCpuAsync const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST EventCpuAsync(EventCpuAsync &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(EventCpuAsync const &) -> EventCpuAsync & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(EventCpuAsync &&) -> EventCpuAsync & = default;
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(EventCpuAsync const & rhs) const
            -> bool
            {
                return (m_spEventCpuAsyncImpl->m_Uuid == rhs.m_spEventCpuAsyncImpl->m_Uuid);
            }
            //-----------------------------------------------------------------------------
            //! Inequality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(EventCpuAsync const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }

        public:
            std::shared_ptr<cpu::detail::EventCpuAsyncImpl> m_spEventCpuAsyncImpl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device event device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                event::EventCpuAsync>
            {
                ALPAKA_FN_HOST static auto getDev(
                    event::EventCpuAsync const & event)
                -> dev::DevCpu
                {
                    return event.m_spEventCpuAsyncImpl->m_Dev;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device event event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                event::EventCpuAsync>
            {
                using type = event::EventCpuAsync;
            };

            //#############################################################################
            //! The CPU device event test trait specialization.
            //#############################################################################
            template<>
            struct EventTest<
                event::EventCpuAsync>
            {
                //-----------------------------------------------------------------------------
                //! \return If the event is not waiting within a stream (not enqueued or already handled).
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto eventTest(
                    event::EventCpuAsync const & event)
                -> bool
                {
                    std::lock_guard<std::mutex> lk(event.m_spEventCpuAsyncImpl->m_Mutex);

                    return event.m_spEventCpuAsyncImpl->m_bIsReady;
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device event thread wait trait specialization.
            //!
            //! Waits until the event itself and therefore all tasks preceding it in the stream it is enqueued to have been completed.
            //! If the event is not enqueued to a stream the method returns immediately.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                event::EventCpuAsync>
            {
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    event::EventCpuAsync const & event)
                -> void
                {
                    wait::wait(event.m_spEventCpuAsyncImpl);
                }
            };
            //#############################################################################
            //! The CPU device event implementation thread wait trait specialization.
            //!
            //! Waits until the event itself and therefore all tasks preceding it in the stream it is enqueued to have been completed.
            //! If the event is not enqueued to a stream the method returns immediately.
            //!
            //! NOTE: This method is for internal usage only.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                std::shared_ptr<event::cpu::detail::EventCpuAsyncImpl>>
            {
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    std::shared_ptr<event::cpu::detail::EventCpuAsyncImpl> const & spEventCpuImpl)
                -> void
                {
                    std::unique_lock<std::mutex> lk(spEventCpuImpl->m_Mutex);

                    if(!spEventCpuImpl->m_bIsReady)
                    {
                        spEventCpuImpl->m_bIsWaitedFor = true;
                        spEventCpuImpl->m_ConditionVariable.wait(
                            lk,
                            [spEventCpuImpl]{return spEventCpuImpl->m_bIsReady;});
                    }
                }
            };
        }
    }
}