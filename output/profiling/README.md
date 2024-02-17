# Profiling Data

## File format

Each file is named based on the method used to generate it. The format is:

```
times_{hash}_{cpu|cuda}_{2x|3x}.{n}.csv
```

Where:
- **hash** is the hash of the git commit of this repository used to build the binary that is run
    - Note: changes to the `run_perf_test.sh` script may be made after the labeled git hash. These changes should not affect the times, merely the format of the output. To reproduce these results, use the commit where the timing files were added to the git repository.
- **cpu/cuda** indicates whether CUDA was enabled during the build step
- **2x/3x** corresponds to the number of simulations
    - 2x uses $2^{n/5}$ where $n$ is from 10 to 82 (inclusive)
    - 3x uses $2^{n/4}$
- **n** is the number for that run; each test is run multiple times to reduce noise
    - "aggregate" means that this is the average of all runs, created using `data_processing.py`

