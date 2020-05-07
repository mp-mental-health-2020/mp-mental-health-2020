import os
import sys
import time

import matplotlib.pyplot as plt


def _get_rgba_color(number_of_distinct_colors, index):
    colors = ["darkorange", "purple", "green", "red", "blue", "yellow"]
    # "nipy_spectral"
    # "tab20b"
    # "viridis"
    # "gnuplot"
    # "brg"
    # color_map = cm.get_cmap("Accent")
    # return color_map(8-index)
    return colors[index]


MARKERS = ['.', 'v', '*', 'p', 'x', 's', 'o', 'd']

# https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
LINE_STYLES = [
    ('solid', 'solid'),
    ('dotted', 'dotted'),
    ('dashed', 'dashed'),
    ('dashdot', 'dashdot'),
    ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
    ('loosely dotted', (0, (1, 10))),
    ('loosely dashed', (0, (5, 10))),
    ('loosely dashdotted', (0, (3, 10, 1, 10))),
]


def visualize_different_sensors(data_frames, number_of_plot_rows, number_of_plot_cols, data_frame_labels=None, dimension_labels=None,
                                axis_labels=None, high_light_points=None, save=False, file_name=None, remove_white_space=True, use_markers=False,
                                use_line_styles=False, align_axes=False):
    # TODO: make use of labels
    # TODO: share axis must be settable from outside
    # TODO: enable shared ticks
    # TODO: Highlight points
    # TODO: mean and std plot
    # plot_classification_over_time holds many useful methods
    fig, axes = plt.subplots(number_of_plot_rows, number_of_plot_cols, sharey='row', figsize=(10, 6))

    try:
        iter(axes)
    except TypeError:
        axes = [axes]

    # setup figure grid
    # gs = gridspec.GridSpec(3, 2)

    # find maximum and minimum over all dimensions in both data frames to limit the y-axis for both plots
    # initial maximum of 0 should work for nearly all sensor and feature readings we are using
    maximum = sys.float_info.min
    minimum = sys.float_info.max

    for df in data_frames:
        maximum = max(maximum, df.apply(max).max())
        minimum = min(minimum, df.apply(min).min())

    spanning_range = abs(maximum - minimum)
    padding = spanning_range * 0.05

    # prepare axes
    a = axes
    if axes.shape[-1] > 1:
        a = axes.flat

    for axis in a:
        # axis.spines['top'].set_visible(False)
        # axis.spines['right'].set_visible(False)
        axis.yaxis.set_ticks_position('left')
        axis.xaxis.set_ticks_position('bottom')

        if align_axes:
            axis.set_ylim([minimum - padding, maximum + padding])

    # horizontal line to indicate threshold
    # fig_left.axhline(y=0, label='Classification Threshold', c="black", alpha=0.5, linestyle='--')

    _plot_data_using_different_styles(data_frames=data_frames, axes=axes, use_markers=use_markers, use_line_styles=use_line_styles)

    # show plot
    if save:
        _save_plot(file_name=file_name, remove_white_space=remove_white_space)
    plt.show()


def _plot_data_using_different_styles(data_frames, axes, use_markers=False, use_line_styles=False):
    if axes.shape[-1] > 1:
        axes = axes.flat
    # plot data in different styles for each column
    for df, axis in zip(data_frames, axes):
        number_of_dimensions = df.shape[1]
        for index in range(number_of_dimensions):
            color = _get_rgba_color(number_of_distinct_colors=number_of_dimensions, index=index)

            marker = None
            line_style = None
            if use_line_styles and len(LINE_STYLES) > number_of_dimensions:
                line_style = LINE_STYLES[index][-1]

            if use_markers and len(MARKERS) > number_of_dimensions:
                marker = MARKERS[index]

            column = df.iloc[:, index]
            column = column.reset_index(drop=True)
            axis.plot(column.index, column.values, color=color, marker=marker, linestyle=line_style)


def _save_plot(file_name, remove_white_space=True):
    project_directory = file_handling.get_project_directory()
    visualization_directory = os.path.join(project_directory, "plots")
    os.makedirs(visualization_directory, exist_ok=True)

    if not file_name:
        file_name = "sample_visualization_{}.png".format(time.time())

    file_name = os.path.join(visualization_directory, file_name)

    if remove_white_space:
        plt.savefig(file_name, bbox_inches='tight')
    else:
        plt.savefig(file_name)
