import preprocessing
from data_reading.phyphox import get_experiments, read_experiment

experiments = get_experiments()
data_frame = read_experiment(experiments[0])
data_frame = preprocessing.align_data(data_frame, listening_rate=20)
# data frame for 5:40 min recording has 16294 data points => 47.9Hz after alignment
# from 5:15-5:18 we have a nice fridge opening
acceleration_x_open_fridge = data_frame['acceleration_z'].iloc[315*48: 318*48].to_numpy()
acceleration_x_open_fridge_2 = data_frame['acceleration_z'].iloc[325*48: 328*48].to_numpy()

## Find the best match with the canonical recursion formula
from dtw import *
alignment = dtw(acceleration_x_open_fridge, acceleration_x_open_fridge_2, keep_internals=True)

## Display the warping curve, i.e. the alignment curve
alignment.plot(type="threeway")

## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
dtw(acceleration_x_open_fridge_2, acceleration_x_open_fridge, keep_internals=True,
    step_pattern=rabinerJuangStepPattern(6, "c"))\
    .plot(type="twoway",offset=-2)

## See the recursion relation, as formula and diagram
print(rabinerJuangStepPattern(6,"c"))
rabinerJuangStepPattern(6,"c").plot()