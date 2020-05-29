import numpy as np

import preprocessing
from data_reading.phyphox import get_experiments, read_experiment

experiments = get_experiments()
data_frame = read_experiment(experiments[0])
data_frame = preprocessing.align_data(data_frame, listening_rate=20)
# data frame for 5:40 min recording has 16294 data points => 47.9Hz after alignment

framerate = 48 # after interpolation we have 48 frames per second

# from 5:15-5:18 we have a nice fridge opening
pattern_start = 315
pattern_end = 320
sensor = 'gyroscope_x'
for s in ["acceleration_x", "acceleration_y", "acceleration_z", "gyroscope_x", "gyroscope_y", "gyroscope_z"]:
    acceleration_x_open_fridge = data_frame[s].iloc[pattern_start*framerate: pattern_end*framerate].to_numpy()
    # another one from 5:25-5:28 we have a nice fridge opening
    acceleration_x_open_fridge_2 = data_frame[s].iloc[2*framerate: 5*framerate].to_numpy()
    ## Find the best match with the canonical recursion formula
    from dtw import *
    alignment = dtw(acceleration_x_open_fridge, acceleration_x_open_fridge_2, keep_internals=True)
    d = alignment.normalizedDistance
    print(d)
acceleration_x_open_fridge = data_frame[sensor].iloc[pattern_start * framerate: pattern_end * framerate].to_numpy()
acceleration_x_open_fridge_2 = data_frame[sensor].iloc[24 * framerate: 27 * framerate].to_numpy()
alignment = dtw(acceleration_x_open_fridge, acceleration_x_open_fridge_2, keep_internals=True)

## Display the warping curve, i.e. the alignment curve
alignment.plot(type="threeway")

## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
"""
res = dtw(acceleration_x_open_fridge_2, acceleration_x_open_fridge, keep_internals=True,
    step_pattern=rabinerJuangStepPattern(6, "c"))
res.plot(type="twoway",offset=-2)
"""
## See the recursion relation, as formula and diagram
# print(rabinerJuangStepPattern(6,"c"))
# rabinerJuangStepPattern(6,"c").plot()

min_dist = 1000
window_step_size = 0.5
sample_length = pattern_end - pattern_start
video_length = len(data_frame) / framerate # video length in seconds

# find the snippets with the smallest distance
similar_patterns = []
for i in range(0,int((video_length-sample_length)/window_step_size)):
    distances = []
    for s in ["acceleration_x", "acceleration_y", "acceleration_z", "gyroscope_x", "gyroscope_y", "gyroscope_z"]:
        acceleration_x_open_fridge = data_frame[s].iloc[pattern_start * framerate: pattern_end * framerate].to_numpy()
        test_sample = data_frame[s].iloc[int(i*window_step_size * framerate): int((i*window_step_size+3/2*sample_length) * framerate)].to_numpy()
        alignment = dtw(acceleration_x_open_fridge, test_sample, keep_internals=True)
        distances.append(alignment.normalizedDistance)
    mean_dist = np.array(distances).mean()
    if all(d < 0.5 for d in distances):
        similar_patterns.append((i*window_step_size, mean_dist))
    if i%20==0:
        print(i*window_step_size)

similar_patterns.sort(key=lambda x:x[1])
similar_patterns