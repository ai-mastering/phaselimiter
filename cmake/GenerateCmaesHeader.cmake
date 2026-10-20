# Keep the upstream submodule untouched. Override only eigenmvn.h in the build
# tree, with checked edits so an upstream update cannot silently drop the fix.
function(phaselimiter_generate_cmaes_header source output)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source}")
  file(READ "${source}" header)
  string(FIND "${header}" "static std::mt19937 rng;" declaration)
  string(FIND "${header}" "    template<typename Scalar>\n      std::mt19937 scalar_normal_dist_op<Scalar>::rng;" definition)
  string(FIND "${header}" "nn,randN)" sampling)
  string(FIND "${header}" "#include <random>" includes)
  if(declaration LESS 0 OR definition LESS 0 OR sampling LESS 0 OR includes LESS 0)
    message(FATAL_ERROR "libcmaes eigenmvn.h changed; review the per-instance RNG patch")
  endif()
  string(REPLACE "#include <random>" "#include <random>\n#include <functional>" header "${header}")
  string(REPLACE "static std::mt19937 rng;" "mutable std::mt19937 rng;" header "${header}")
  string(REPLACE "    template<typename Scalar>\n      std::mt19937 scalar_normal_dist_op<Scalar>::rng;"
    "    // RNG storage belongs to each Gaussian sampler." header "${header}")
  # Eigen copies expression functors. Refer to the owning sampler so evaluations
  # advance its engine AND normal_distribution cache across successive batches.
  string(REPLACE "nn,randN)" "nn,std::ref(randN))" header "${header}")
  file(CONFIGURE OUTPUT "${output}" CONTENT "${header}" @ONLY)
endfunction()
