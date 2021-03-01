#!/bin/bash

rm -f /tmp/cppcheck_failed
(find src \
| grep -e "\.\(h\|cpp\)$" \
| grep -v "miniz\.h" \
| grep -v "miniz\.cpp" \
| grep -v "duk_config\.h" \
| grep -v "duktape\.h" \
| grep -v "fftsg\.cpp" \
| grep -v "spectrum_distribution_base_image\.cpp" \
| grep -v "GradCalculator\.h" \
| xargs \
cppcheck \
  -j `nproc` \
  --error-exitcode=1 \
  --force \
  --std=c++11 \
  --language=c++ \
  --enable=warning,performance,style \
  --suppressions-list=cppcheck_suppressions.txt \
  --xml 2>/tmp/results/cppcheck_err.xml) || (echo failed > /tmp/cppcheck_failed)
mkdir -p /tmp/results/cppcheck
cppcheck-htmlreport --file=/tmp/results/cppcheck_err.xml --report-dir=/tmp/results/cppcheck --source-dir=.
if [ -f /tmp/cppcheck_failed ]; then
    echo 'cppcheck failed'
    exit 1
fi
echo 'cppcheck passed'
