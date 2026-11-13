"""
This module contains a class containing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from typing import List, Dict, Optional, Tuple, Union, Callable, Any
import seaborn as sns
from datetime import datetime
import logging
from config.logger_config import LoggerMixin


class GenericDataVisualizer(LoggerMixin):
    """
    A flexible, generic plotting class for visualizing various types of data.
    Provides functionality for comparison plots, time series visualization,
    and error analysis with highly customizable components.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 color_palette: Optional[str] = 'viridis',
                 verbose: bool = False):
        """
        Initialize the plotting class with data.

        :param data: DataFrame containing the data to be visualized
        :param color_palette: Matplotlib/Seaborn color palette to use
        :param verbose: set logger level to INFO

        """
        self.data = data.copy()
        self.color_palette = color_palette

        if verbose:
            self.logger.setLevel(logging.INFO)
            self.logger.info(f"Initialized with DataFrame of shape {self.data.shape}")
            self.logger.info(f"Columns: {list(self.data.columns)}")
        else:
            self.logger.setLevel(logging.WARNING)

    def comparison_plot(self,
                        x_column: str,
                        y_columns: Union[str, List[str]],
                        reference_column: Optional[str] = None,
                        component_columns: Optional[List[str]] = None,
                        plot_type: str = 'scatter',
                        title: str = 'Data Comparison',
                        figsize: Tuple[int, int] = (12, 8),
                        x_label: Optional[str] = None,
                        y_label: Optional[str] = None,
                        legend_loc: str = 'best',
                        style_kwargs: Optional[Dict] = None,
                        ax: Optional[plt.Axes] = None,
                        add_trendline: bool = False,
                        add_reference_line: bool = False,
                        reference_line_style: str = '1:1',  # '1:1' or 'best-fit'
                        add_percent_bands: bool = False,
                        percent_band_values: List[float] = [10, 20],
                        band_colors: Optional[List[str]] = None,
                        band_alphas: Optional[List[float]] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a flexible comparison plot between different data columns.

        :param x_column: name of the column in data that should be plotted
        :param y_columns: list of column names that should be plotted
        :param reference_column: Optional reference column to compare against (e.g., measured data)
        :param component_columns: Optional list of component columns that make up the y_column
        :param plot_type: type of plot ('scatter', 'line', or 'both')
        :param title: title of the plot
        :param figsize: size of the figure
        :param x_label: label of the x axis
        :param y_label: label of the y axis
        :param legend_loc: location of the legend on the plot
        :param style_kwargs: keyword arguments to pass to the plotting function
        :param add_trendline: add a trendline to the plot
        :param add_reference_line: add a reference line to the plot
        :param reference_line_style: type of reference line ('1:1' or 'best-fit')
        :param add_percent_bands: add percentage difference bands around the reference line
        :param percent_band_values: list of percentage values to use for bands (e.g., [10, 20] for ±10% and ±20%)
        :param band_colors: optional list of colors for the bands
        :param band_alphas: optional list of alpha values for band transparency

        Returns:
            Tuple of (figure, axes)
        """
        # Validate input columns
        for col in [x_column]:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")

        # Handle y_columns as either string or list
        if isinstance(y_columns, str):
            y_columns = [y_columns]

        for col in y_columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")

        if reference_column and reference_column not in self.data.columns:
            raise ValueError(f"Reference column '{reference_column}' not found in data")

        if component_columns:
            for col in component_columns:
                if col not in self.data.columns:
                    raise ValueError(f"Component column '{col}' not found in data")

        # Create figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        # Set default styles
        base_alpha = 0.8
        base_scatter_size = 25
        base_line_width = 2

        # Process style overrides
        if style_kwargs:
            if 'alpha' in style_kwargs:
                base_alpha = style_kwargs['alpha']
            if 's' in style_kwargs:
                base_scatter_size = style_kwargs['s']
            elif 'markersize' in style_kwargs:
                # Convert markersize to s (approximation)
                base_scatter_size = style_kwargs['markersize'] ** 2
            if 'linewidth' in style_kwargs:
                base_line_width = style_kwargs['linewidth']

        # Determine number of colors needed
        n_colors = len(y_columns) + (1 if reference_column else 0) + (
            len(component_columns) if component_columns else 0)
        colors = sns.color_palette(self.color_palette, n_colors)
        color_idx = 0

        # Plot main y columns
        for col in y_columns:
            if plot_type == 'scatter' or plot_type == 'both':
                ax.scatter(self.data[x_column], self.data[col],
                           label=col if plot_type == 'scatter' else None,
                           color=colors[color_idx], s=base_scatter_size, alpha=base_alpha)

            if plot_type == 'line' or plot_type == 'both':
                ax.plot(self.data[x_column], self.data[col],
                        label=col if plot_type == 'line' else None,
                        color=colors[color_idx], linewidth=base_line_width, alpha=base_alpha)

            color_idx += 1

            # Add trend line if requested
            if add_trendline:
                try:
                    mask = ~(np.isnan(self.data[x_column]) | np.isnan(self.data[col]))
                    if mask.sum() > 1:  # Need at least 2 points for a line
                        z = np.polyfit(self.data[x_column][mask], self.data[col][mask], 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(self.data[x_column][mask].min(), self.data[x_column][mask].max(), 100)
                        ax.plot(x_trend, p(x_trend), '--', color=colors[color_idx - 1], alpha=0.6,
                                linewidth=base_line_width / 1.5)
                except Exception as e:
                    self.logger.warning(f"Could not add trend line: {str(e)}")

        # Plot reference column if provided
        if reference_column:
            if plot_type == 'scatter' or plot_type == 'both':
                ax.scatter(self.data[x_column], self.data[reference_column],
                           label=reference_column if plot_type == 'scatter' else None,
                           color=colors[color_idx], marker='x', s=base_scatter_size * 2, alpha=base_alpha)

            if plot_type == 'line' or plot_type == 'both':
                ax.plot(self.data[x_column], self.data[reference_column],
                        label=reference_column if plot_type == 'line' else None,
                        color=colors[color_idx], linestyle='--', linewidth=base_line_width * 1.5, alpha=base_alpha)

            color_idx += 1

        # Plot component columns if provided
        if component_columns:
            for col in component_columns:
                if plot_type == 'scatter' or plot_type == 'both':
                    ax.scatter(self.data[x_column], self.data[col],
                               label=f"{col}" if plot_type == 'scatter' else None,
                               color=colors[color_idx], s=base_scatter_size * 0.6, alpha=base_alpha * 0.8)

                if plot_type == 'line' or plot_type == 'both':
                    ax.plot(self.data[x_column], self.data[col],
                            label=f"{col}" if plot_type == 'line' else None,
                            color=colors[color_idx], linewidth=base_line_width * 0.8, alpha=base_alpha * 0.8)

                color_idx += 1

        # Determine reference line and bands
        if add_reference_line:
            # Prepare x values for reference line
            # If we're comparing to a reference column, use the min/max from both
            if reference_column and len(y_columns) == 1:
                min_val = min(self.data[reference_column].min(), self.data[y_columns[0]].min())
                max_val = max(self.data[reference_column].max(), self.data[y_columns[0]].max())
            else:
                # Otherwise use the min/max from x_column and the first y_column
                min_val = min(self.data[x_column].min(), self.data[y_columns[0]].min())
                max_val = max(self.data[x_column].max(), self.data[y_columns[0]].max())

            line_vals = np.linspace(min_val, max_val, 100)

            # Create reference line
            if reference_line_style == '1:1':
                # Simple 1:1 line
                ax.plot(line_vals, line_vals, 'k--', alpha=0.5, label='1:1 Line')

                # Add percentage bands if requested
                if add_percent_bands and len(percent_band_values) > 0:
                    # Default colors and alphas if not provided
                    if band_colors is None:
                        band_colors = ['r', 'g', 'b', 'c', 'm'][:len(percent_band_values)]

                    if band_alphas is None:
                        band_alphas = [0.1] * len(percent_band_values)

                    # Ensure we have enough colors and alphas
                    band_colors = band_colors[:len(percent_band_values)]
                    band_alphas = band_alphas[:len(percent_band_values)]

                    for i, percent in enumerate(percent_band_values):
                        # Calculate upper and lower bound values
                        upper_vals = line_vals * (1 + percent / 100)
                        lower_vals = line_vals * (1 - percent / 100)

                        # Plot the bands
                        ax.fill_between(line_vals, lower_vals, upper_vals,
                                        color=band_colors[i], alpha=band_alphas[i],
                                        label=f'±{percent}%')

                        # Also plot the band lines for clarity
                        ax.plot(line_vals, upper_vals, '--', color=band_colors[i],
                                linewidth=1, alpha=0.5)
                        ax.plot(line_vals, lower_vals, '--', color=band_colors[i],
                                linewidth=1, alpha=0.5)

            elif reference_line_style == 'best-fit':
                # Create a best-fit line between x_column and first y_column
                try:
                    mask = ~(np.isnan(self.data[x_column]) | np.isnan(self.data[y_columns[0]]))
                    if mask.sum() > 1:  # Need at least 2 points for a line
                        z = np.polyfit(self.data[x_column][mask], self.data[y_columns[0]][mask], 1)
                        p = np.poly1d(z)
                        x_ref = np.linspace(self.data[x_column][mask].min(), self.data[x_column][mask].max(), 100)
                        y_ref = p(x_ref)
                        ax.plot(x_ref, y_ref, 'k--', alpha=0.5,
                                label=f'Best Fit (y = {z[0]:.3f}x + {z[1]:.3f})')

                        # Add percentage bands around the best-fit line if requested
                        if add_percent_bands and len(percent_band_values) > 0:
                            # Default colors and alphas if not provided
                            if band_colors is None:
                                band_colors = ['r', 'g', 'b', 'c', 'm'][:len(percent_band_values)]

                            if band_alphas is None:
                                band_alphas = [0.1] * len(percent_band_values)

                            # Ensure we have enough colors and alphas
                            band_colors = band_colors[:len(percent_band_values)]
                            band_alphas = band_alphas[:len(percent_band_values)]

                            for i, percent in enumerate(percent_band_values):
                                # Calculate absolute offset based on percentage of y values
                                offset = percent / 100 * np.abs(y_ref)

                                # Plot the bands
                                ax.fill_between(x_ref, y_ref - offset, y_ref + offset,
                                                color=band_colors[i], alpha=band_alphas[i],
                                                label=f'±{percent}%')

                                # Also plot the band lines for clarity
                                ax.plot(x_ref, y_ref + offset, '--', color=band_colors[i],
                                        linewidth=1, alpha=0.5)
                                ax.plot(x_ref, y_ref - offset, '--', color=band_colors[i],
                                        linewidth=1, alpha=0.5)
                except Exception as e:
                    self.logger.warning(f"Could not add best-fit reference line: {str(e)}")

        # Set labels and title
        ax.set_xlabel(x_label if x_label else x_column)
        ax.set_ylabel(y_label if y_label else "Value")
        ax.set_title(title)
        ax.legend(loc=legend_loc)
        ax.grid(True, alpha=0.3)

        return fig, ax

    def time_series_plot(self,
                         time_column: str,
                         y_columns: Union[str, List[str]],
                         secondary_y_columns: Optional[Union[str, List[str]]] = None,
                         component_columns: Optional[List[str]] = None,
                         start_time: Optional[Union[str, datetime]] = None,
                         end_time: Optional[Union[str, datetime]] = None,
                         figsize: Tuple[int, int] = (14, 8),
                         title: str = 'Time Series Plot',
                         y_label: Optional[str] = None,
                         secondary_y_label: Optional[str] = None,
                         style_kwargs: Optional[Dict] = None,
                         date_format: str = '%Y-%m-%d %H:%M',
                         twin_y_axis: bool = True,
                         fill_area: Union[bool, Dict[str, bool]] = False,
                         fill_alpha: float = 0.3,
                         ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a flexible time series plot with optional secondary axis.

        :param time_column: Name of the time column.
        :param y_columns: Main column(s) to plot on primary y-axis (can be a single column name or a list)
        :param secondary_y_columns: Optional column(s) to plot on secondary y-axis
        :param component_columns: Optional list of component columns to include on primary axis
        :param start_time: Optional start time filter
        :param end_time: Optional end time filter
        :param figsize: Figure size as (width, height)
        :param title: Plot title.
        :param y_label: Label of the y axis.
        :param secondary_y_label: Label of the secondary y axis.
        :param style_kwargs: Keyword arguments to pass to the plot style.
        :param date_format: Date format of the time series plot.
        :param twin_y_axis: Twin axis of the time series plot.
        :param ax: Axis of the time series plot.
        :param fill_area: Whether to fill area under curves. Can be:
                         - bool: applies to all lines
                         - dict: maps column names to bool values for selective filling
        :param fill_alpha: Alpha transparency for filled areas (0-1)
        :return: Figure and Axis of the time series plot.

        Returns:
            Tuple of (figure, axes) or (figure, (primary_axis, secondary_axis))
        """
        # Validate input columns
        if time_column not in self.data.columns:
            raise ValueError(f"Time column '{time_column}' not found in data")

        # Handle y_columns as either string or list
        if isinstance(y_columns, str):
            y_columns = [y_columns]

        for col in y_columns:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")

        # Handle secondary_y_columns
        if secondary_y_columns:
            if isinstance(secondary_y_columns, str):
                secondary_y_columns = [secondary_y_columns]

            for col in secondary_y_columns:
                if col not in self.data.columns:
                    raise ValueError(f"Secondary column '{col}' not found in data")

        # Check component columns
        if component_columns:
            for col in component_columns:
                if col not in self.data.columns:
                    raise ValueError(f"Component column '{col}' not found in data")

        # Create a copy of data to filter
        plot_data = self.data.copy()

        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(plot_data[time_column]):
            try:
                plot_data[time_column] = pd.to_datetime(plot_data[time_column])
                self.logger.info(f"Converted {time_column} to datetime")
            except Exception as e:
                raise ValueError(f"Could not convert {time_column} to datetime: {str(e)}")

        # Filter by time range if specified
        if start_time:
            if isinstance(start_time, str):
                start_time = pd.to_datetime(start_time)

            # Handle timezone mismatch
            if plot_data[time_column].dt.tz is not None and start_time.tz is None:
                # Data has timezone, start_time doesn't - add timezone to start_time
                start_time = start_time.tz_localize(plot_data[time_column].dt.tz)
            elif plot_data[time_column].dt.tz is None and start_time.tz is not None:
                # start_time has timezone, data doesn't - remove timezone from start_time
                start_time = start_time.tz_localize(None)

            plot_data = plot_data[plot_data[time_column] >= start_time]

        if end_time:
            if isinstance(end_time, str):
                end_time = pd.to_datetime(end_time)

            # Handle timezone mismatch
            if plot_data[time_column].dt.tz is not None and end_time.tz is None:
                # Data has timezone, end_time doesn't - add timezone to end_time
                end_time = end_time.tz_localize(plot_data[time_column].dt.tz)
            elif plot_data[time_column].dt.tz is None and end_time.tz is not None:
                # end_time has timezone, data doesn't - remove timezone from end_time
                end_time = end_time.tz_localize(None)

            plot_data = plot_data[plot_data[time_column] <= end_time]

        # Create figure if not provided
        if ax is None:
            fig, ax1 = plt.subplots(figsize=figsize)
        else:

            #debug
            if isinstance(ax, np.ndarray):
                raise ValueError(f"Expected single Axes object, got numpy array: {ax}")
            if not hasattr(ax, 'figure'):
                raise ValueError(f"ax parameter must be a matplotlib Axes object, got {type(ax)}")

            ax1 = ax
            fig = ax1.figure

        # Set default styles
        base_alpha = 0.8
        base_line_width = 2
        base_marker_size = 4
        marker = 'o'

        # Apply style overrides if provided
        if style_kwargs:
            if 'alpha' in style_kwargs:
                base_alpha = style_kwargs['alpha']
            if 'linewidth' in style_kwargs:
                base_line_width = style_kwargs['linewidth']
            if 'markersize' in style_kwargs:
                base_marker_size = style_kwargs['markersize']
            if 'marker' in style_kwargs:
                marker = style_kwargs['marker']

        # Helper function to determine if a column should be filled
        def should_fill(column_name):
            if isinstance(fill_area, bool):
                return fill_area
            elif isinstance(fill_area, dict):
                return fill_area.get(column_name, False)
            else:
                return False

        # Get color palette
        n_colors = len(y_columns) + (len(secondary_y_columns) if secondary_y_columns else 0) + (
            len(component_columns) if component_columns else 0)
        colors = sns.color_palette(self.color_palette, n_colors)
        color_idx = 0

        # Plot primary y columns
        for col in y_columns:
            color = colors[color_idx]
            ax1.plot(plot_data[time_column], plot_data[col],
                     label=col, color=colors[color_idx],
                     linewidth=base_line_width, alpha=base_alpha,
                     marker=marker, markersize=base_marker_size)
            color_idx += 1

        # Plot component columns if provided
        if component_columns:
            for col in component_columns:
                ax1.plot(plot_data[time_column], plot_data[col],
                         label=col, color=colors[color_idx],
                         linewidth=base_line_width * 0.8, alpha=base_alpha * 0.8,
                         marker=marker, markersize=base_marker_size * 0.8)

                # Fill area if requested
                if should_fill(col):
                    ax1.fill_between(plot_data[time_column], plot_data[col],
                                     alpha=fill_alpha, color=color)

                color_idx += 1

        # Set up primary axis labels
        ax1.set_xlabel('Time')
        ax1.set_ylabel(y_label if y_label else "Value")
        ax1.grid(True, alpha=0.3)

        # Format x-axis to show dates nicely
        ax1.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        plt.xticks(rotation=45)

        # Plot secondary y columns if provided
        if secondary_y_columns and twin_y_axis:
            ax2 = ax1.twinx()

            for col in secondary_y_columns:
                ax2.plot(plot_data[time_column], plot_data[col],
                         label=col, color=colors[color_idx],
                         linestyle='-.', linewidth=base_line_width, alpha=base_alpha,
                         marker=marker, markersize=base_marker_size)

                # Fill area if requested
                if should_fill(col):
                    ax2.fill_between(plot_data[time_column], plot_data[col],
                                     alpha=fill_alpha, color=color)

                color_idx += 1


            ax2.set_ylabel(secondary_y_label if secondary_y_label else "Secondary Value")

            # Create combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        else:
            # Add secondary columns to primary axis if twin_y_axis is False
            if secondary_y_columns:
                for col in secondary_y_columns:
                    ax1.plot(plot_data[time_column], plot_data[col],
                             label=col, color=colors[color_idx],
                             linestyle='-.', linewidth=base_line_width, alpha=base_alpha,
                             marker=marker, markersize=base_marker_size)

                    # Fill area if requested
                    if should_fill(col):
                        ax1.fill_between(plot_data[time_column], plot_data[col],
                                         alpha=fill_alpha, color=color)

                    color_idx += 1

            ax1.legend(loc='best')
            ax2 = None

        plt.title(title)
        plt.tight_layout()

        if ax2:
            return fig, (ax1, ax2)
        else:
            return fig, ax1

    def error_analysis_plot(self,
                            predicted_column: str,
                            actual_column: str,
                            x_column: Optional[str] = None,
                            figsize: Tuple[int, int] = (16, 10),
                            title: str = 'Prediction Error Analysis',
                            style_kwargs: Optional[Dict] = None,
                            error_thresholds_pct: Union[float, List[float]] = 10,
                            threshold_colors: Optional[List[str]] = None,
                            threshold_alphas: Optional[List[float]] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a multi-panel plot for analyzing prediction errors.

        Args:
            predicted_column: Column containing predicted/calculated values
            actual_column: Column containing actual/measured values
            x_column: Optional column to plot error against (e.g., time, speed)
            figsize: Figure size as (width, height)
            title: Title for the plot
            style_kwargs: Additional styling parameters

        Returns:
            Tuple of (figure, axes)
        """
        # Validate input columns
        # Validate input columns
        if predicted_column not in self.data.columns:
            raise ValueError(f"Predicted column '{predicted_column}' not found in data")

        if actual_column not in self.data.columns:
            raise ValueError(f"Actual column '{actual_column}' not found in data")

        if x_column and x_column not in self.data.columns:
            raise ValueError(f"X column '{x_column}' not found in data")

        # Handle error_thresholds_pct as either float or list
        if isinstance(error_thresholds_pct, (int, float)):
            error_thresholds_pct = [float(error_thresholds_pct)]

        # Sort thresholds from smallest to largest
        error_thresholds_pct = sorted(error_thresholds_pct)

        # Default colors and alphas
        if threshold_colors is None:
            threshold_colors = ['green', 'blue', 'orange', 'purple', 'brown'][:len(error_thresholds_pct)]
        if threshold_alphas is None:
            threshold_alphas = [0.3] * len(error_thresholds_pct)

        # Ensure we have enough colors and alphas
        threshold_colors = threshold_colors[:len(error_thresholds_pct)]
        threshold_alphas = threshold_alphas[:len(error_thresholds_pct)]

        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16)

        # Calculate error metrics
        error = self.data[predicted_column] - self.data[actual_column]
        percent_error = (error / self.data[actual_column]) * 100

        # Add error columns to data for reference
        self.data['error'] = error
        self.data['percent_error'] = percent_error

        # Set default styles
        base_alpha = 0.7
        base_scatter_size = 36

        # Apply custom styling if provided
        if style_kwargs:
            if 'alpha' in style_kwargs:
                base_alpha = style_kwargs['alpha']
            if 's' in style_kwargs:
                base_scatter_size = style_kwargs['s']
            elif 'markersize' in style_kwargs:
                base_scatter_size = style_kwargs['markersize'] ** 2

        # Plot 1: Actual vs Predicted scatter with 1:1 line
        axs[0, 0].scatter(self.data[actual_column], self.data[predicted_column],
                          alpha=base_alpha, s=base_scatter_size)

        # Add 1:1 line
        min_val = min(self.data[actual_column].min(), self.data[predicted_column].min())
        max_val = max(self.data[actual_column].max(), self.data[predicted_column].max())
        line_vals = np.linspace(min_val, max_val, 100)
        axs[0, 0].plot(line_vals, line_vals, 'k--', alpha=0.5)

        # Add percentage bands to the scatter plot
        for i, threshold in enumerate(error_thresholds_pct):
            # Calculate upper and lower bounds
            upper_vals = line_vals * (1 + threshold / 100)
            lower_vals = line_vals * (1 - threshold / 100)

            # Fill between the curves
            axs[0, 0].fill_between(line_vals, lower_vals, upper_vals,
                                   color=threshold_colors[i], alpha=threshold_alphas[i] * 0.5,
                                   label=f'±{threshold}%')

            # Add boundary lines
            axs[0, 0].plot(line_vals, upper_vals, '--', color=threshold_colors[i],
                           linewidth=1, alpha=0.6)
            axs[0, 0].plot(line_vals, lower_vals, '--', color=threshold_colors[i],
                           linewidth=1, alpha=0.6)

        axs[0, 0].set_xlabel(f'Actual ({actual_column})')
        axs[0, 0].set_ylabel(f'Predicted ({predicted_column})')
        axs[0, 0].set_title('Actual vs Predicted')
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend(loc='best')

        # Try to add correlation info
        try:
            correlation = np.corrcoef(self.data[actual_column].dropna(), self.data[predicted_column].dropna())[0, 1]
            r_squared = correlation ** 2
            axs[0, 0].text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=axs[0, 0].transAxes,
                           verticalalignment='top', bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.7})
        except:
            pass

        # Plot 2: Error distribution histogram
        bins = min(20, len(self.data) // 5)  # Adaptive number of bins
        axs[0, 1].hist(error.dropna(), bins=max(5, bins), alpha=base_alpha)
        axs[0, 1].axvline(0, color='red', linestyle='--')
        axs[0, 1].set_xlabel('Error')
        axs[0, 1].set_ylabel('Frequency')
        axs[0, 1].set_title('Error Distribution')

        # Add mean and std text
        mean_err = error.mean()
        std_err = error.std()
        axs[0, 1].text(0.05, 0.95, f'Mean: {mean_err:.3f}\nStd: {std_err:.3f}',
                       transform=axs[0, 1].transAxes, verticalalignment='top',
                       bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.7})

        # Plot 3: Error vs X column (or actual values if x_column not provided)
        x_data = self.data[x_column] if x_column else self.data[actual_column]
        x_label = x_column if x_column else actual_column

        axs[1, 0].scatter(x_data, error, alpha=base_alpha, s=base_scatter_size)
        axs[1, 0].axhline(0, color='red', linestyle='--')
        axs[1, 0].set_xlabel(x_label)
        axs[1, 0].set_ylabel('Error')
        axs[1, 0].set_title(f'Error vs {x_label}')
        axs[1, 0].grid(True, alpha=0.3)

        # Try to add a trend line if enough data points
        if len(self.data) >= 3:
            try:
                mask = ~(np.isnan(x_data) | np.isnan(error))
                if mask.sum() > 1:  # Need at least 2 points for a line
                    z = np.polyfit(x_data[mask], error[mask], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(x_data[mask].min(), x_data[mask].max(), 100)
                    axs[1, 0].plot(x_trend, p(x_trend), 'r-', alpha=0.5)
            except:
                pass

        # Plot 4: Percent Error vs X column
        axs[1, 1].scatter(x_data, percent_error, alpha=base_alpha, s=base_scatter_size)
        axs[1, 1].axhline(0, color='red', linestyle='--')

        # Add threshold lines with custom colors
        for i, threshold in enumerate(error_thresholds_pct):
            axs[1, 1].axhline(threshold, color=threshold_colors[i], linestyle='--', alpha=0.5)
            axs[1, 1].axhline(-threshold, color=threshold_colors[i], linestyle='--', alpha=0.5)

            # Add shaded regions for each threshold
            axs[1, 1].axhspan(-threshold, threshold, color=threshold_colors[i], alpha=threshold_alphas[i] * 0.3)

        axs[1, 1].set_xlabel(x_label)
        axs[1, 1].set_ylabel('Percent Error (%)')
        axs[1, 1].set_title(f'Percent Error vs {x_label}')
        axs[1, 1].grid(True, alpha=0.3)

        # Add overall metrics text box
        mean_pct_err = percent_error.mean()
        rmse = np.sqrt(np.mean(error ** 2))

        # Calculate % of points within each threshold
        within_threshold_stats = []
        for threshold in error_thresholds_pct:
            within_pct = (abs(percent_error) < threshold).mean() * 100
            within_threshold_stats.append(f'Values within ±{threshold}%: {within_pct:.1f}%')

        # Join the stats with a ' | ' separator
        threshold_stats_text = ' | '.join(within_threshold_stats)

        # Add the metrics text box
        fig.text(0.5, 0.01,
                 f'RMSE: {rmse:.3f} | Mean % Error: {mean_pct_err:.2f}% | {threshold_stats_text}',
                 ha='center', fontsize=12,
                 bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.7})

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.12)

        return fig, axs

    def bar_chart(self,
                  x_column: str,
                  y_columns: Union[str, List[str]],
                  secondary_y_columns: Optional[Union[str, List[str]]] = None,
                  group_by: Optional[str] = None,
                  orientation: str = 'vertical',
                  figsize: Tuple[int, int] = (12, 8),
                  title: str = 'Bar Chart',
                  x_label: Optional[str] = None,
                  y_label: Optional[str] = None,
                  secondary_y_label: Optional[str] = None,
                  legend_loc: str = 'best',
                  style_kwargs: Optional[Dict] = None,
                  bar_width: float = 0.8,
                  show_values: bool = False,
                  value_format: str = '{:.1f}',
                  add_totals: bool = False,
                  add_error_bars: bool = False,
                  error_columns: Optional[Union[str, List[str]]] = None,
                  stacked: bool = False,
                  sort_bars: Optional[str] = None,  # 'ascending', 'descending', or None
                  ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]]:
        """
        Create a flexible bar chart with support for multiple y-axes, grouping, and various customization options.

        :param x_column: Column name to use for x-axis categories
        :param y_columns: Main column(s) to plot as bars on primary y-axis (can be a single column name or a list)
        :param secondary_y_columns: Optional column(s) to plot on secondary y-axis
        :param group_by: Optional column to group bars by (creates grouped bar charts)
        :param orientation: 'vertical' or 'horizontal' bars
        :param figsize: Figure size as (width, height)
        :param title: Plot title
        :param x_label: Label for x-axis
        :param y_label: Label for primary y-axis
        :param secondary_y_label: Label for secondary y-axis
        :param legend_loc: Location of the legend
        :param style_kwargs: Additional styling parameters
        :param bar_width: Width of the bars (between 0 and 1)
        :param show_values: Whether to display values on top of bars
        :param value_format: Format string for displayed values
        :param add_totals: Add a "Total" bar that sums across all categories
        :param add_error_bars: Add error bars to the bars
        :param error_columns: Column(s) containing error values for error bars
        :param stacked: Whether to stack bars (only applies when multiple y_columns)
        :param sort_bars: Optional sorting of bars ('ascending', 'descending', or None)
        :param ax: Optional existing axes to plot on

        :return: Tuple of (figure, axes) or (figure, (primary_axis, secondary_axis))
        """
        # Validate input columns
        if x_column not in self.data.columns:
            raise ValueError(f"X column '{x_column}' not found in data")

        # Handle y_columns as either string or list
        if isinstance(y_columns, str):
            y_columns = [y_columns]

        for col in y_columns:
            if col not in self.data.columns:
                raise ValueError(f"Y column '{col}' not found in data")

        # Handle secondary_y_columns
        if secondary_y_columns:
            if isinstance(secondary_y_columns, str):
                secondary_y_columns = [secondary_y_columns]

            for col in secondary_y_columns:
                if col not in self.data.columns:
                    raise ValueError(f"Secondary Y column '{col}' not found in data")

        # Handle group_by
        if group_by and group_by not in self.data.columns:
            raise ValueError(f"Group by column '{group_by}' not found in data")

        # Handle error columns
        if add_error_bars:
            if error_columns is None:
                raise ValueError("Error columns must be provided when add_error_bars is True")

            if isinstance(error_columns, str):
                error_columns = [error_columns]

            # Check if we have the correct number of error columns
            expected_error_cols = len(y_columns)
            if secondary_y_columns:
                expected_error_cols += len(secondary_y_columns)
            if group_by and not stacked:
                expected_error_cols *= len(self.data[group_by].unique())

            if len(error_columns) != expected_error_cols:
                raise ValueError(
                    f"Number of error columns ({len(error_columns)}) must match expected number ({expected_error_cols})")

            for col in error_columns:
                if col not in self.data.columns:
                    raise ValueError(f"Error column '{col}' not found in data")

        # Create figure if not provided
        if ax is None:
            fig, ax1 = plt.subplots(figsize=figsize)
        else:
            ax1 = ax
            fig = ax1.figure

        # Create a copy of data for manipulation
        plot_data = self.data.copy()

        # Create the secondary axis if needed
        ax2 = ax1.twinx() if secondary_y_columns else None

        # Set default style parameters
        bar_alpha = 0.8
        edge_color = 'black'
        edge_width = 0.5

        # Apply custom styling if provided
        if style_kwargs:
            if 'alpha' in style_kwargs:
                bar_alpha = style_kwargs['alpha']
            if 'edgecolor' in style_kwargs:
                edge_color = style_kwargs['edgecolor']
            if 'linewidth' in style_kwargs:
                edge_width = style_kwargs['linewidth']

        # Get unique x values, optionally sorted
        x_categories = plot_data[x_column].unique()

        # Sort bars if requested
        if sort_bars:
            # For sorting, we'll use the sum of y_columns values for each x category
            if group_by:
                sort_data = plot_data.groupby(x_column)[y_columns].sum().sum(axis=1)
            else:
                sort_data = plot_data.groupby(x_column)[y_columns].sum().sum(axis=1)

            if sort_bars == 'ascending':
                sorted_indices = sort_data.sort_values().index
            elif sort_bars == 'descending':
                sorted_indices = sort_data.sort_values(ascending=False).index
            else:
                raise ValueError(f"Invalid sort_bars value: {sort_bars}. Use 'ascending', 'descending', or None")
            x_categories = np.array([x for x in sorted_indices if x in x_categories])

        # Prepare colors for the bars
        n_colors = len(y_columns)
        if group_by and not stacked:
            n_colors = len(plot_data[group_by].unique())
        elif group_by and stacked:
            n_colors = len(y_columns)
        colors = sns.color_palette(self.color_palette, n_colors)

        # Add totals if requested
        if add_totals and not group_by:
            # Create a summary row with totals across all categories
            totals = plot_data.sum(numeric_only=True).to_frame().T
            totals[x_column] = 'Total'
            plot_data = pd.concat([plot_data, totals], ignore_index=True)
            x_categories = np.append(x_categories, 'Total')

        # Initialize variables for the plotting logic
        bar_positions = {}
        all_bars = []
        all_labels = []

        # Case 1: No grouping
        if not group_by:
            # Calculate bar positions
            x_indices = np.arange(len(x_categories))

            if not stacked:
                # Side-by-side bars for multiple y columns
                width = bar_width / len(y_columns)
                for i, col in enumerate(y_columns):
                    # Offset for positioning side-by-side bars
                    offset = i * width - bar_width / 2 + width / 2
                    positions = x_indices + offset

                    # Get values for this column across all x categories
                    y_values = []
                    for cat in x_categories:
                        cat_data = plot_data[plot_data[x_column] == cat]
                        y_values.append(cat_data[col].sum() if len(cat_data) > 0 else 0)

                    # Plot the bars with appropriate orientation
                    if orientation == 'vertical':
                        bars = ax1.bar(positions, y_values, width=width, color=colors[i],
                                       alpha=bar_alpha, edgecolor=edge_color, linewidth=edge_width,
                                       label=col)

                        # Add error bars if requested
                        if add_error_bars and i < len(error_columns):
                            error_values = []
                            for cat in x_categories:
                                cat_data = plot_data[plot_data[x_column] == cat]
                                error_values.append(cat_data[error_columns[i]].sum() if len(cat_data) > 0 else 0)
                            ax1.errorbar(positions, y_values, yerr=error_values, fmt='none',
                                         ecolor='black', capsize=3)

                        # Add value labels on top of bars if requested
                        if show_values:
                            for j, v in enumerate(y_values):
                                if v != 0:  # Only show non-zero values
                                    ax1.text(positions[j], v, value_format.format(v),
                                             ha='center', va='bottom', fontsize=8)
                    else:  # Horizontal
                        bars = ax1.barh(positions, y_values, height=width, color=colors[i],
                                        alpha=bar_alpha, edgecolor=edge_color, linewidth=edge_width,
                                        label=col)

                        # Add error bars if requested
                        if add_error_bars and i < len(error_columns):
                            error_values = []
                            for cat in x_categories:
                                cat_data = plot_data[plot_data[x_column] == cat]
                                error_values.append(cat_data[error_columns[i]].sum() if len(cat_data) > 0 else 0)
                            ax1.errorbar(y_values, positions, xerr=error_values, fmt='none',
                                         ecolor='black', capsize=3)

                        # Add value labels to the right of bars if requested
                        if show_values:
                            for j, v in enumerate(y_values):
                                if v != 0:  # Only show non-zero values
                                    ax1.text(v, positions[j], value_format.format(v),
                                             ha='left', va='center', fontsize=8)

                    all_bars.append(bars)
                    all_labels.append(col)
                    bar_positions[col] = positions
            else:
                # Stacked bars for multiple y columns
                bottom = np.zeros(len(x_categories))

                for i, col in enumerate(y_columns):
                    y_values = []
                    for cat in x_categories:
                        cat_data = plot_data[plot_data[x_column] == cat]
                        y_values.append(cat_data[col].sum() if len(cat_data) > 0 else 0)

                    y_values = np.array(y_values)

                    if orientation == 'vertical':
                        bars = ax1.bar(x_indices, y_values, bottom=bottom, width=bar_width,
                                       color=colors[i], alpha=bar_alpha,
                                       edgecolor=edge_color, linewidth=edge_width, label=col)

                        # Add error bars on top of the stack
                        if add_error_bars and i < len(error_columns):
                            error_values = []
                            for cat in x_categories:
                                cat_data = plot_data[plot_data[x_column] == cat]
                                error_values.append(cat_data[error_columns[i]].sum() if len(cat_data) > 0 else 0)
                            # Error bars go on top of current stack
                            ax1.errorbar(x_indices, bottom + y_values, yerr=error_values, fmt='none',
                                         ecolor='black', capsize=3, alpha=0.7)

                        # Add value labels in the middle of stacked sections if requested
                        if show_values:
                            for j, v in enumerate(y_values):
                                if v > 0:  # Only show if the value is positive
                                    ax1.text(x_indices[j], bottom[j] + v / 2, value_format.format(v),
                                             ha='center', va='center', fontsize=8, fontweight='bold',
                                             color='white' if bar_alpha > 0.7 else 'black')
                    else:  # Horizontal
                        bars = ax1.barh(x_indices, y_values, left=bottom, height=bar_width,
                                        color=colors[i], alpha=bar_alpha,
                                        edgecolor=edge_color, linewidth=edge_width, label=col)

                        # Add error bars on top of the stack
                        if add_error_bars and i < len(error_columns):
                            error_values = []
                            for cat in x_categories:
                                cat_data = plot_data[plot_data[x_column] == cat]
                                error_values.append(cat_data[error_columns[i]].sum() if len(cat_data) > 0 else 0)
                            # Error bars go on right side of current stack
                            ax1.errorbar(bottom + y_values, x_indices, xerr=error_values, fmt='none',
                                         ecolor='black', capsize=3, alpha=0.7)

                        # Add value labels in the middle of stacked sections if requested
                        if show_values:
                            for j, v in enumerate(y_values):
                                if v > 0:  # Only show if the value is positive
                                    ax1.text(bottom[j] + v / 2, x_indices[j], value_format.format(v),
                                             ha='center', va='center', fontsize=8, fontweight='bold',
                                             color='white' if bar_alpha > 0.7 else 'black')

                    bottom += y_values
                    all_bars.append(bars)
                    all_labels.append(col)

                bar_positions['stacked'] = x_indices

            # Plot secondary y columns if provided (as lines for stacked, points for regular)
            if secondary_y_columns:
                for i, col in enumerate(secondary_y_columns):
                    y_values = []
                    for cat in x_categories:
                        cat_data = plot_data[plot_data[x_column] == cat]
                        y_values.append(cat_data[col].sum() if len(cat_data) > 0 else 0)

                    # Use different marker or style for secondary axis to distinguish
                    secondary_color = colors[min(len(y_columns) + i, len(colors) - 1)]

                    if orientation == 'vertical':
                        line = ax2.plot(x_indices, y_values, 'o-', color=secondary_color,
                                        linewidth=2, markersize=6, label=col)

                        # Add error bars if requested
                        if add_error_bars and (len(y_columns) + i) < len(error_columns):
                            error_values = []
                            for cat in x_categories:
                                cat_data = plot_data[plot_data[x_column] == cat]
                                error_values.append(
                                    cat_data[error_columns[len(y_columns) + i]].sum() if len(cat_data) > 0 else 0)
                            ax2.errorbar(x_indices, y_values, yerr=error_values, fmt='none',
                                         ecolor=secondary_color, alpha=0.7, capsize=3)
                    else:  # Horizontal
                        line = ax2.plot(y_values, x_indices, 'o-', color=secondary_color,
                                        linewidth=2, markersize=6, label=col)

                        # Add error bars if requested
                        if add_error_bars and (len(y_columns) + i) < len(error_columns):
                            error_values = []
                            for cat in x_categories:
                                cat_data = plot_data[plot_data[x_column] == cat]
                                error_values.append(
                                    cat_data[error_columns[len(y_columns) + i]].sum() if len(cat_data) > 0 else 0)
                            ax2.errorbar(y_values, x_indices, xerr=error_values, fmt='none',
                                         ecolor=secondary_color, alpha=0.7, capsize=3)

                    all_bars.append(line[0])
                    all_labels.append(col)

        # Case 2: Grouped by another column
        else:
            # Get unique groups and x categories
            groups = sorted(plot_data[group_by].unique())
            x_indices = np.arange(len(x_categories))

            if not stacked:
                # Traditional grouped bars
                width = bar_width / len(groups)

                # For each y-column, create grouped bars
                for y_idx, y_col in enumerate(y_columns):
                    for g_idx, group in enumerate(groups):
                        # Filter data for this group
                        group_data = plot_data[plot_data[group_by] == group]

                        # Calculate positions for this group's bars
                        offset = g_idx * width - bar_width / 2 + width / 2
                        positions = x_indices + offset

                        # Get values for all x categories in this group
                        y_values = []
                        for cat in x_categories:
                            cat_group_data = group_data[group_data[x_column] == cat]
                            y_values.append(cat_group_data[y_col].sum() if len(cat_group_data) > 0 else 0)

                        # Create grouped bars with appropriate orientation
                        if orientation == 'vertical':
                            bars = ax1.bar(positions, y_values, width=width,
                                           color=colors[g_idx], alpha=bar_alpha,
                                           edgecolor=edge_color, linewidth=edge_width,
                                           label=f"{group}" if y_idx == 0 else "")

                            # Add value labels if requested
                            if show_values:
                                for j, v in enumerate(y_values):
                                    if v != 0:
                                        ax1.text(positions[j], v, value_format.format(v),
                                                 ha='center', va='bottom', fontsize=8)
                        else:  # Horizontal
                            bars = ax1.barh(positions, y_values, height=width,
                                            color=colors[g_idx], alpha=bar_alpha,
                                            edgecolor=edge_color, linewidth=edge_width,
                                            label=f"{group}" if y_idx == 0 else "")

                            # Add value labels if requested
                            if show_values:
                                for j, v in enumerate(y_values):
                                    if v != 0:
                                        ax1.text(v, positions[j], value_format.format(v),
                                                 ha='left', va='center', fontsize=8)

                        if y_idx == 0:  # Only add to legend once per group
                            all_bars.append(bars)
                            all_labels.append(f"{group}")

                        bar_positions[f"{group}_{y_col}"] = positions
            else:
                # Stacked grouped bars - each group gets its own stacked bar
                width = bar_width / len(groups)

                for g_idx, group in enumerate(groups):
                    # Calculate positions for this group's bars
                    offset = g_idx * width - bar_width / 2 + width / 2
                    positions = x_indices + offset

                    # Initialize bottom for stacking within this group
                    bottom = np.zeros(len(x_categories))

                    # Stack all y_columns for this group
                    for y_idx, y_col in enumerate(y_columns):
                        # Filter data for this group
                        group_data = plot_data[plot_data[group_by] == group]

                        # Get values for all x categories in this group
                        y_values = []
                        for cat in x_categories:
                            cat_group_data = group_data[group_data[x_column] == cat]
                            y_values.append(cat_group_data[y_col].sum() if len(cat_group_data) > 0 else 0)

                        y_values = np.array(y_values)

                        # Create stacked bars for this group
                        if orientation == 'vertical':
                            bars = ax1.bar(positions, y_values, bottom=bottom, width=width,
                                           color=colors[y_idx], alpha=bar_alpha,
                                           edgecolor=edge_color, linewidth=edge_width,
                                           label=f"{y_col}" if g_idx == 0 else "")

                            # Add value labels if requested
                            if show_values:
                                for j, v in enumerate(y_values):
                                    if v > 0:
                                        ax1.text(positions[j], bottom[j] + v / 2, value_format.format(v),
                                                 ha='center', va='center', fontsize=7, fontweight='bold',
                                                 color='white' if bar_alpha > 0.7 else 'black')
                        else:  # Horizontal
                            bars = ax1.barh(positions, y_values, left=bottom, height=width,
                                            color=colors[y_idx], alpha=bar_alpha,
                                            edgecolor=edge_color, linewidth=edge_width,
                                            label=f"{y_col}" if g_idx == 0 else "")

                            # Add value labels if requested
                            if show_values:
                                for j, v in enumerate(y_values):
                                    if v > 0:
                                        ax1.text(bottom[j] + v / 2, positions[j], value_format.format(v),
                                                 ha='center', va='center', fontsize=7, fontweight='bold',
                                                 color='white' if bar_alpha > 0.7 else 'black')

                        if g_idx == 0:  # Only add to legend once per y_column
                            all_bars.append(bars)
                            all_labels.append(f"{y_col}")

                        bottom += y_values

        # Set x-axis tick labels based on orientation
        if orientation == 'vertical':
            ax1.set_xticks(x_indices)
            ax1.set_xticklabels(x_categories, rotation=45 if len(x_categories) > 5 else 0,
                                ha='right' if len(x_categories) > 5 else 'center')
        else:  # Horizontal
            ax1.set_yticks(x_indices)
            ax1.set_yticklabels(x_categories)

        # Set labels and title
        if orientation == 'vertical':
            ax1.set_xlabel(x_label if x_label else x_column)
            ax1.set_ylabel(y_label if y_label else "Value")
            if secondary_y_columns and ax2:
                ax2.set_ylabel(secondary_y_label if secondary_y_label else "Secondary Value")
        else:  # Horizontal
            ax1.set_ylabel(x_label if x_label else x_column)
            ax1.set_xlabel(y_label if y_label else "Value")
            if secondary_y_columns and ax2:
                ax2.set_xlabel(secondary_y_label if secondary_y_label else "Secondary Value")

        ax1.set_title(title)
        ax1.grid(True, axis='y' if orientation == 'vertical' else 'x', alpha=0.3)

        # Create combined legend for primary and secondary axis
        if len(all_bars) > 0:
            ax1.legend(all_bars, all_labels, loc=legend_loc)

        # Adjust layout
        plt.tight_layout()

        # Return appropriate axes tuple
        if secondary_y_columns and ax2:
            return fig, (ax1, ax2)
        else:
            return fig, ax1

