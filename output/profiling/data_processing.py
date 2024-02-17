#!/usr/bin/env python
import glob
import sys

args = sys.argv[1:]
if len(args) != 3:
    print("""
Usage:
    python data_processing.py HASH DEVICE SCENE_SCALING
""")
    exit(1)

hash = args[0]
device = args[1]
scene_scaling = args[2]

# CPU data
files = glob.glob(f'./times_{hash}_{device}_{scene_scaling}.[0-9].csv')
files = [open(file) for file in files]
file_lines = [file.readlines() for file in files]
for file in files:
    file.close()

with open(f"times_{hash}_{device}_{scene_scaling}.aggregate.csv", "w") as f:
    kernel = "Kernel"
    if device == "cpu":
        kernel = "Non-setup"
    f.write(f"Scene Count,{device.upper()} {kernel} Time (s),{device.upper()} Simulation Time (s)\n")
    for line_index in range(1, len(file_lines[0])):
        output = [0,0,0]
        for lines in file_lines:
            line = lines[line_index]
            [scene_count, kernel_time, total_time] = line.split(",")
            output[0] = int(scene_count)
            output[1] += int(kernel_time)
            output[2] += int(total_time)
        # Average and convert to seconds
        output[1] = output[1] / len(files) / 1_000_000_000
        output[2] = output[2] / len(files) / 1_000_000_000
        f.write(f"{output[0]},{output[1]},{output[2]}\n")

