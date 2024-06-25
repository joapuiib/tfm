!#/usr/bin/env python3

import math

original = 15855
crop_size = 1812
n_samples = 9
shift = math.ceil((original - crop_size) / (n_samples - 1))
print(f"shift: {shift}")
print(f"overlap: {crop_size - shift}")

x = 0
y = 0
for i in range(n_samples):
    print(f"crop {i} ({x}, {y}, {x + crop_size}, {y + crop_size})")
    x += shift
    if x + crop_size > original:
        x = original - crop_size
    # y += padding
