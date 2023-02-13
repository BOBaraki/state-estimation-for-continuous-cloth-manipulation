"""
==============
Eventplot Demo
==============

An eventplot showing sequences of events with various line properties.
The plot is shown in both horizontal and vertical orientations.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd

import pdb

matplotlib.rcParams['font.size'] = 8.0

# Fixing random state for reproducibility
np.random.seed(19680801)


# create random data
# data1 = np.random.random([3, 50])
df = pd.read_csv('UR_82.csv')
pdb.set_trace()
gt = df['label_idx'].to_numpy()
predictions = df['predictions'].to_numpy()



# gt[gt == 3] = 4
# gt[gt == 2] = 6
# gt[gt == 1] = 2
#
# predictions[predictions == 3] = 3
# predictions[predictions == 2] = 5
# predictions[predictions == 1] = 1


# import pdb
# pdb.set_trace()
gt[gt == 2] = 8
gt[gt == 3] = 6
gt[gt == 1] = 4
gt[gt == 0] = 2




# import pdb
# pdb.set_trace()
predictions[predictions == 2] = 7
predictions[predictions == 3] = 5
predictions[predictions == 1] = 3
predictions[predictions == 0] = 1




# print(np.shape(predictions))

timeframe = np.arange(len(gt))

data1 = np.vstack((predictions, gt))

# fig, ax = plt.subplots()
#
#
# ax.plot(timeframe, predictions, label="predictions")
# ax.plot(timeframe, gt, label="gt")
# ax.legend()
#
#
# plt.show()

# import pdb
# pdb.set_trace()
# set different colors for each set of positions
# colors1 = ['C{}'.format(i) for i in range(2)]
# #
# # # set different line properties for each set of positions
# # # note that some overlap
# lineoffsets1 = np.array([0, 1]) #topothesia value sto aksona y
# linelengths1 = [1, 1]  #platos
# #
# fig, axs = plt.subplots(2)
# #
# # # create a horizontal plot
# axs[0].eventplot(data1, colors=colors1, lineoffsets=lineoffsets1,
#                     linelengths=linelengths1)
# #
# # # create a vertical plot
# axs[1].eventplot(data1, colors=colors1, lineoffsets=lineoffsets1,
#                     linelengths=linelengths1, orientation='vertical')
#
# plt.show()

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource

color_gt = []
color_pred = []

# for i in range(len(gt)):
#     if gt[i] == 6:
#         color_gt.append("red")
#     elif gt[i] == 4:
#         color_gt.append("blue")
#     elif gt[i] == 2:
#         color_gt.append("yellow")
#
# for i in range(len(gt)):
#     if predictions[i] == 5:
#         color_pred.append("red")
#     elif predictions[i] == 3:
#         color_pred.append("blue")
#     elif predictions[i] == 1:
#         color_pred.append("yellow")

pdb.set_trace()

for i in range(len(gt)):
    if gt[i] == 8:
        color_gt.append("red")
    elif gt[i] == 6:
        color_gt.append("blue")
    elif gt[i] == 4:
        color_gt.append("yellow")
    elif gt[i] == 2:
        color_gt.append("green")

for i in range(len(gt)):
    if predictions[i] == 7:
        color_pred.append("red")
    elif predictions[i] == 5:
        color_pred.append("blue")
    elif predictions[i] == 3:
        color_pred.append("yellow")
    elif predictions[i] == 1:
        color_pred.append("green")

# import pdb
# pdb.set_trace()

df['color_gt'] = color_gt
df['color_pred'] = color_pred

df['newgt'] = gt
df['newpred'] = predictions


# import pdb
# pdb.set_trace()

p = figure(
    width=1000,
    height=1000,
    title="Predictions vs GT",
    toolbar_location="above",
)

p.rect(
    source=ColumnDataSource(df),
    x="index",
    y='newgt',
    width=0.01,
    height=0.4,
    fill_color="color_gt",
    line_color="color_gt",
)

p.rect(
    source=ColumnDataSource(df),
    x="index",
    y='newpred',
    width=0.01,
    height=0.4,
    fill_color="color_pred",
    line_color="color_pred",
)

show(p)
#
# logger.close()

# create another set of random data.
# the gamma distribution is only used fo aesthetic purposes
# data2 = np.random.gamma(4, size=[60, 50])
#
# # use individual values for the parameters this time
# # these values will be used for all data sets (except lineoffsets2, which
# # sets the increment between each data set in this usage)
# colors2 = 'black'
# lineoffsets2 = 1
# linelengths2 = 1
#
# # create a horizontal plot
# axs[0, 1].eventplot(data2, colors=colors2, lineoffsets=lineoffsets2,
#                     linelengths=linelengths2)
#
#
# # create a vertical plot
# axs[1, 1].eventplot(data2, colors=colors2, lineoffsets=lineoffsets2,
#                     linelengths=linelengths2, orientation='vertical')

