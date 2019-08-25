#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "clSPARSE" for configuration "Release"
set_property(TARGET clSPARSE APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clSPARSE PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/libclSPARSE.so.0.10.2.0"
  IMPORTED_SONAME_RELEASE "libclSPARSE.so.1"
  )

list(APPEND _IMPORT_CHECK_TARGETS clSPARSE )
list(APPEND _IMPORT_CHECK_FILES_FOR_clSPARSE "${_IMPORT_PREFIX}/lib64/libclSPARSE.so.0.10.2.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
