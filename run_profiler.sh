cmake --build build/release_cuda --parallel -t performance

# sudo ncu --kernel-name kernel --launch-skip 0 --launch-count 1 -o output/$(git log --oneline | head -n 1 | awk '{ print $1 }') --import-source on --section SourceCounters --section Occupancy "./build/release_cuda/tests/performance" -s 21504 -m 0
time sudo ncu --kernel-name kernel --launch-skip 0 --launch-count 1 -o output/$(git log --oneline | head -n 1 | awk '{ print $1 }') --import-source on --set full "./build/release_cuda/tests/performance" -s 21504 -m 0

