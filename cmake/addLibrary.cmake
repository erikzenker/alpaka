################################################################################
# Copyright 2015-2016 Benjamin Worpitz, Maximilian Knespel
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
# RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE
# USE OR PERFORMANCE OF THIS SOFTWARE.
################################################################################

# CUDA_SOURCE_PROPERTY_FORMAT is only supported starting from 3.3.0.
CMAKE_MINIMUM_REQUIRED(VERSION 3.3.0)

#------------------------------------------------------------------------------
# Calls CUDA_ADD_LIBRARY or ADD_LIBRARY depending on the enabled alpaka
# accelerators.
#
# ALPAKA_ADD_LIBRARY( cuda_target file0 file1 ... [STATIC | SHARED | MODULE]
#   [EXCLUDE_FROM_ALL] [OPTIONS <nvcc-flags> ... ] )
#
# In order to be compliant with both ADD_LIBRARY and CUDA_ADD_LIBRARY
# the position of STATIC, SHARED, MODULE, EXCLUDE_FROM_ALL options don't matter.
# This also means you won't be able to include files with those exact same
# case-sensitive names.
# After OPTIONS only nvcc compiler flags are allowed though. And for readiblity
# and portability you shouldn't completely mix STATIC, ... with the source
# code filenames!
# OPTIONS and the arguments thereafter are ignored if not using CUDA, they
# won't throw an error in that case.
#------------------------------------------------------------------------------
FUNCTION(ALPAKA_ADD_LIBRARY libraryName)
    # CUDA_ADD_LIBRARY( cuda_target file0 file1 ...
    #                   [STATIC | SHARED | MODULE]
    #                   [EXCLUDE_FROM_ALL] [OPTIONS <nvcc-flags> ... ] )
    # add_library( <name> [STATIC | SHARED | MODULE]
    #              [EXCLUDE_FROM_ALL]
    #              source1 [source2 ...] )

    # traverse arguments and sort them by option and source files
    SET( arguments ${ARGN} )
    SET( optionsEncountered OFF )
    UNSET( libraryType )
    UNSET( excludeFromAll )
    UNSET( optionArguments )
    FOREACH( argument IN LISTS arguments )
        # 1.) check for OPTIONS
        IF( argument STREQUAL "OPTIONS" )
            IF ( optionsEncountered )
                MESSAGE( FATAL_ERROR "[ALPAKA_ADD_LIBRARY] OPTIONS subcommand specified more than one time. This is not allowed!" )
            ELSE()
                SET( optionsEncountered ON )
            ENDIF()
        ENDIF()

        # 2.) check if inside OPTIONS, because then all other checks are
        # unnecessary although they could give hints about wrong locations
        # of those subcommands
        IF( optionsEncountered )
            LIST( APPEND optionArguments "${argument}" )
            CONTINUE()
        ENDIF()

        # 3.) check for libraryType and EXCLUDE_FROM_ALL
        IF( ( argument STREQUAL "STATIC" ) OR
            ( argument STREQUAL "SHARED" ) OR
            ( argument STREQUAL "MODULE" )
        )
            IF( DEFINED libraryType )
                message( FATAL_ERROR "Setting more than one library type option ( STATIC SHARED MODULE ) not allowed!" )
            ENDIF()
            set( libraryType ${argument} )
            CONTINUE()
        ENDIF()
        IF( argument STREQUAL "EXCLUDE_FROM_ALL" )
            SET( excludeFromAll ${argument} )
            CONTINUE()
        ENDIF()

        # 4.) ELSE the argument is a file name
        list( APPEND sourceFileNames "${argument}" )
    ENDFOREACH()
    UNSET( optionsEncountered )
    #message( "libraryType = ${libraryType}" )
    #message( "sourceFileNames = ${sourceFileNames}" )

    # call add_library or cuda_add_library now
    IF( ALPAKA_ACC_GPU_CUDA_ENABLE )
        FOREACH( _file ${ARGN} )
            IF( ( ${_file} MATCHES "\\.cpp$" ) OR
                ( ${_file} MATCHES "\\.cxx$" )
            )
                SET_SOURCE_FILES_PROPERTIES( ${_file} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )
            ENDIF()
        ENDFOREACH()
        CMAKE_POLICY(SET CMP0023 OLD)   # CUDA_ADD_LIBRARY calls TARGET_LINK_LIBRARIES without keywords.
        CUDA_ADD_LIBRARY(
            ${libraryName}
            ${sourceFileNames}
            ${libraryType}
            ${excludeFromAll}
            ${optionArguments}
        )
    ELSE()
        #message( "add_library( ${libraryName} ${libraryType} ${excludeFromAll} ${sourceFileNames} )" )
        ADD_LIBRARY(
            ${libraryName}
            ${libraryType}
            ${excludeFromAll}
            ${sourceFileNames}
        )
    ENDIF()

    # UNSET variables (not sure if necessary)
    UNSET( libraryType )
    UNSET( sourceFileNames )
    UNSET( excludeFromAll )
    UNSET( optionArguments )
ENDFUNCTION()