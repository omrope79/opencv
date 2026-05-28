# Sphinx wrapper for the OpenCV doc/ tree.
# Adds a `sphinx` custom target so `cmake --build <build> --target sphinx`
# works the same way `--target doxygen` does.
#
# Gated on BUILD_DOCS so it follows the existing toggle.

if(NOT BUILD_DOCS)
  return()
endif()

# Prefer the sphinx-build that lives in the conda env where myst-parser /
# pydata-sphinx-theme / breathe / exhale are installed. Override on the
# CMake command line:  -DSPHINX_BUILD=/path/to/sphinx-build
find_program(SPHINX_BUILD
  NAMES sphinx-build
  HINTS /home/abhishek/miniconda3/envs/opencvpy310/bin
  DOC "Path to sphinx-build")

if(NOT SPHINX_BUILD)
  message(STATUS "docs_sphinx: sphinx-build not found; `sphinx` target disabled")
  return()
endif()

set(_SPHINX_CONFDIR    "${CMAKE_CURRENT_SOURCE_DIR}")
set(_SPHINX_OUTDIR     "${CMAKE_CURRENT_BINARY_DIR}/html")
set(_SPHINX_INPUT_ROOT "${CMAKE_CURRENT_BINARY_DIR}/docs_sphinx_input")

# Stage a single Sphinx srcdir at ${CMAKE_BINARY_DIR}/docs_sphinx_input/.
# Symlinks tutorials and enabled contrib modules without modifying original trees.
set(_SPHINX_INPUT_TUTORIALS "${_SPHINX_INPUT_ROOT}/tutorials")
set(_SPHINX_INPUT_CONTRIB   "${_SPHINX_INPUT_ROOT}/tutorials_contrib")
file(REMOVE_RECURSE "${_SPHINX_INPUT_ROOT}")
file(MAKE_DIRECTORY "${_SPHINX_INPUT_TUTORIALS}")

# Main tree: symlink the master file + each main module subtree.
file(CREATE_LINK
  "${CMAKE_SOURCE_DIR}/doc/tutorials/tutorials.markdown"
  "${_SPHINX_INPUT_TUTORIALS}/tutorials.markdown"
  SYMBOLIC COPY_ON_ERROR)
file(GLOB _main_tutorial_children
     RELATIVE "${CMAKE_SOURCE_DIR}/doc/tutorials"
     "${CMAKE_SOURCE_DIR}/doc/tutorials/*")
foreach(_d ${_main_tutorial_children})
  if(IS_DIRECTORY "${CMAKE_SOURCE_DIR}/doc/tutorials/${_d}")
    file(CREATE_LINK
      "${CMAKE_SOURCE_DIR}/doc/tutorials/${_d}"
      "${_SPHINX_INPUT_TUTORIALS}/${_d}"
      SYMBOLIC COPY_ON_ERROR)
  endif()
endforeach()

# Stage contrib modules with a tutorials/ dir if OPENCV_EXTRA_MODULES_PATH is set.
# Selection list lives in conf.py's CONTRIB_MODULES.
if(OPENCV_EXTRA_MODULES_PATH AND EXISTS "${OPENCV_EXTRA_MODULES_PATH}")
  file(MAKE_DIRECTORY "${_SPHINX_INPUT_CONTRIB}")
  set(_contrib_root_md "${_SPHINX_INPUT_CONTRIB}/contrib_root.markdown")
  file(WRITE "${_contrib_root_md}"
    "Tutorials for contrib modules {#tutorial_contrib_root}\n"
    "=============================\n\n")
  file(GLOB _contrib_subdirs RELATIVE "${OPENCV_EXTRA_MODULES_PATH}"
       "${OPENCV_EXTRA_MODULES_PATH}/*")
  foreach(_m ${_contrib_subdirs})
    set(_tut_dir "${OPENCV_EXTRA_MODULES_PATH}/${_m}/tutorials")
    if(IS_DIRECTORY "${_tut_dir}")
      file(CREATE_LINK "${_tut_dir}" "${_SPHINX_INPUT_CONTRIB}/${_m}"
           SYMBOLIC COPY_ON_ERROR)
      # Match every top-level *.markdown in <m>/tutorials/ directly.
      file(GLOB _tocs RELATIVE "${_tut_dir}" "${_tut_dir}/*.markdown")
      foreach(_t ${_tocs})
        file(STRINGS "${_tut_dir}/${_t}" _id LIMIT_COUNT 1 REGEX ".*\\{#[^}]+\\}")
        string(REGEX REPLACE ".*\\{#([^}]+)\\}.*" "\\1" _id "${_id}")
        if(_id)
          file(APPEND "${_contrib_root_md}" "- ${_m}. @subpage ${_id}\n")
        endif()
      endforeach()
    endif()
  endforeach()
endif()

# API reference: Generate Doxygen XML by overriding legacy Doxyfile format flags.
# The legacy Doxyfile exists because doc configures before docs_sphinx.
set(_LEGACY_DOXYFILE "${CMAKE_BINARY_DIR}/doc/Doxyfile")
set(_DOXYGEN_XML_DIR "${CMAKE_BINARY_DIR}/doc/doxygen/xml")
if(EXISTS "${_LEGACY_DOXYFILE}")
  set(_SPHINX_DOXYFILE "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile-xml")
  file(WRITE "${_SPHINX_DOXYFILE}"
    "@INCLUDE = ${_LEGACY_DOXYFILE}\n"
    "GENERATE_HTML = NO\n"
    "GENERATE_LATEX = NO\n"
    "GENERATE_XML = YES\n"
    "XML_OUTPUT = xml\n"
    "XML_PROGRAMLISTING = NO\n"
    # Force flat XML output for Breathe compatibility (disables CREATE_SUBDIRS).
    "CREATE_SUBDIRS = NO\n"
    # Exclude Objective-C headers to prevent Breathe crashes down the line.
    "EXCLUDE_PATTERNS += cap_ios.h\n"
    # Disable graph generation as we don't load sphinx.ext.graphviz.
    "CLASS_GRAPH = NO\n"
    "COLLABORATION_GRAPH = NO\n"
    "GROUP_GRAPHS = NO\n"
    "INCLUDE_GRAPH = NO\n"
    "INCLUDED_BY_GRAPH = NO\n"
    "DIRECTORY_GRAPH = NO\n"
  )
  find_program(DOXYGEN_EXE NAMES doxygen)
  if(DOXYGEN_EXE)
    # sphinx-xml target regenerates Doxygen XML; runs automatically with `sphinx`.
    add_custom_target(sphinx-xml
      COMMAND ${CMAKE_COMMAND} -E make_directory ${_DOXYGEN_XML_DIR}
      COMMAND ${DOXYGEN_EXE} ${_SPHINX_DOXYFILE}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/doc
      COMMENT "Doxygen XML (for Sphinx breathe) -> ${_DOXYGEN_XML_DIR}"
      VERBATIM)
    message(STATUS "docs_sphinx: sphinx-xml target enabled (doxygen: ${DOXYGEN_EXE})")
  else()
    message(STATUS "docs_sphinx: doxygen not found; `sphinx-xml` disabled")
  endif()

  # API stubs generation relies on parsing Doxygen XML.
  # This is deferred to sphinx-build logic within conf.py.
else()
  message(STATUS "docs_sphinx: legacy Doxyfile not present at ${_LEGACY_DOXYFILE}; "
                 "`sphinx-xml` disabled (re-run cmake after doc/ subdir configures)")
endif()

# Pass staging options via env. Override CONTRIB_MODULES at your shell manually.
set(_SPHINX_ENV
  "OPENCV_SPHINX_INPUT_ROOT=${_SPHINX_INPUT_ROOT}"
  "OPENCV_CONTRIB_ROOT=${OPENCV_EXTRA_MODULES_PATH}"
  "OPENCV_DOXYGEN_XML_DIR=${_DOXYGEN_XML_DIR}"
)

# Parallel knob. Sphinx supports `-j auto` (=nproc), `-j N`, or a literal
# integer; override per-build with -DSPHINX_JOBS=8. `auto` is the sweet spot
# on developer machines — using nproc workers takes the API-reference build
# from minutes to seconds. conf.py declares `parallel_read_safe`/
# `parallel_write_safe`, and breathe ≥ 4.30 is parallel-safe, so this is
# unconditionally enabled. `--keep-going` keeps the build progressing past
# non-fatal warnings (we still see them in the log).
if(NOT SPHINX_JOBS)
  set(SPHINX_JOBS "auto")
endif()
set(_SPHINX_WARNINGS "${CMAKE_CURRENT_BINARY_DIR}/sphinx-warnings.log")
add_custom_target(sphinx
  COMMAND ${CMAKE_COMMAND} -E make_directory ${_SPHINX_OUTDIR}
  COMMAND ${CMAKE_COMMAND} -E env ${_SPHINX_ENV}
          ${SPHINX_BUILD} -j ${SPHINX_JOBS}
                          --keep-going
                          -w ${_SPHINX_WARNINGS}
                          -c ${_SPHINX_CONFDIR}
                          ${_SPHINX_INPUT_ROOT}
                          ${_SPHINX_OUTDIR}
  WORKING_DIRECTORY ${_SPHINX_CONFDIR}
  COMMENT "Building Sphinx HTML site -> ${_SPHINX_OUTDIR} (jobs=${SPHINX_JOBS})"
  VERBATIM)

# Ensure breathe sees current XML on every `sphinx` build.
if(TARGET sphinx-xml)
  add_dependencies(sphinx sphinx-xml)
endif()

# Clean rebuild: `cmake --build <build> --target sphinx-clean`
add_custom_target(sphinx-clean
  COMMAND ${CMAKE_COMMAND} -E rm -rf ${_SPHINX_OUTDIR}
  COMMENT "Removing ${_SPHINX_OUTDIR}"
  VERBATIM)

message(STATUS "docs_sphinx: sphinx target enabled (sphinx-build: ${SPHINX_BUILD})")
