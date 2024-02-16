#!/bin/bash
#

file=output/prof/times_$(git log --oneline | head -n 1 | awk '{ print $1 }')_autogen.csv

echo Building...

cmake --build build/release -t performance

echo Testing...
echo

echo "scene count\tsimulation time\ttotal time" | tee $file

for sim_count in {10..82}
  do
    sim_count=$(calc 'int(2^('$sim_count'/5))' | awk '{print $1}')
    ./build/release/tests/performance -m 0 -s $sim_count -a gp \
      | grep "Kernel took:" \
      | awk '{print("'$sim_count', ",$3,", ",$6)}' \
      | tee -a $file | awk -F ',' '{print $1,"\t",$2,"\t",$3}' \
      || (echo "Error! $sim_count"; exit 1)
  done

echo
echo Done
