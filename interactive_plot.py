# interactive_colorplot.py

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

class InteractiveColorPlot:
    def __init__(self, data_list, x_list=None, y_list=None, use_slider=True):
        """
        Parameters:
        - data_list: list of 2D arrays, can have different shapes
        - x_list, y_list: optional list of 1D arrays for axes
        """
        self.data_list = data_list
        self.num_datasets = len(data_list)
        self.x_list = x_list if x_list is not None else [np.arange(data.shape[1]) for data in data_list]
        self.y_list = y_list if y_list is not None else [np.arange(data.shape[0]) for data in data_list]
        self.use_slider = use_slider
        self.saved_points = {i: [] for i in range(self.num_datasets)}
        self.im = None
        self._init_plot()
        self._init_widget()

    def _init_plot(self):
        self.fig, (self.axc, self.axs) = plt.subplots(2, 1, figsize=(10, 15))
        self.current_index = 0
        # scatter plot on the colour plot
        self.scatter_plot = self.axc.scatter([], [], c='r', s=40, marker='.', zorder=2)
        # scatter plot by itself
        self.scatter2 = self.axs.scatter([], [], c='r', s=40, marker='.')
        self._plot_current_dataset()
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _plot_current_dataset(self):
        data = self.data_list[self.current_index]
        x = self.x_list[self.current_index]
        y = self.y_list[self.current_index]

        # Create X, Y meshgrids from x, y centers
        X, Y = np.meshgrid(x, y)

        # Remove previous colormesh if it exists
        if self.im:
            self.im.remove()

        self.im = self.axc.pcolormesh(X, Y, np.transpose(data), shading='auto', cmap='viridis')
        self.axc.set_title(f"Dataset {self.current_index}")
        self.axc.set_xlim(x[-1], x[0])
        self.axc.set_ylim(y[0], y[-1])
        self.fig.canvas.draw_idle()

        # limits on the 2nd scatter plot
        self.axs.set_xlim(x[0], x[-1])
        self.axs.set_ylim(0, 20e9)
        
        # update scatter plot
        self._update_scatter()


    def _init_widget(self):
        if self.use_slider:
            self.widget = widgets.IntSlider(value=0, min=0, max=self.num_datasets - 1, step=1, description='Dataset:')
        else:
            self.widget = widgets.Dropdown(options=[(f"Dataset {i}", i) for i in range(self.num_datasets)],
                                           value=0, description='Dataset:')
        self.widget.observe(self._update_dataset, names='value')
        display(self.widget)

    def _update_dataset(self, change):
        self.current_index = change['new']
        self._plot_current_dataset()
        self._update_scatter()


    def _on_click(self, event):
        if event.inaxes == self.axc:
            x, y = event.xdata, event.ydata
            self.saved_points[self.current_index].append((x, y))
            self._update_scatter()

    def _update_scatter(self):
        points = np.array(self.saved_points[self.current_index])
        if points.size == 0:
            self.scatter_plot.set_offsets(np.empty((0, 2)))
        else:
            self.scatter_plot.set_offsets(points)
        # Combine all points across datasets for the second subplot
        all_points = []
        for pts in self.saved_points.values():
            all_points.extend(pts)
        all_points = np.array(all_points)

        if all_points.size == 0:
            self.scatter2.set_offsets(np.empty((0, 2)))
        else:
            self.scatter2.set_offsets(all_points)
            self.fig.canvas.draw_idle()

    def get_saved_points(self, dataset_index=None):
        """Return saved points. If dataset_index is None, returns dict for all."""
        if dataset_index is None:
            return self.saved_points
        return self.saved_points.get(dataset_index, [])

    def clear_saved_points(self, dataset_index=None):
        """Clear saved points. If dataset_index is None, clears all."""
        if dataset_index is None:
            for k in self.saved_points:
                self.saved_points[k] = []
        else:
            self.saved_points[dataset_index] = []
        self._update_scatter()