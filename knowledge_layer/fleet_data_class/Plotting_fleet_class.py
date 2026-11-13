"""
A class which contains functions for plotting fleet data
"""

import pandas as pd
import numpy as np

import math
import matplotlib.pyplot as plt
from scipy import stats

from scipy.stats import anderson

from config.loggermixin.logger_config import LoggerMixin


class FleetPlottingClass(LoggerMixin):

    def __init__(self,
                 debug:bool=False,
                 verbose:bool=True):
        self.debug = debug
        self.verbose = verbose

    @staticmethod
    def wrap_title(title: str,
                   max_chars: int = 40) -> str:
        """
        Wrap plot titles if they exceed a certain character length.
        Tries to break at natural points like 'vs' or before second set of variables.

        Args:
            title: The title string to wrap
            max_chars: Maximum characters before wrapping (default 40)

        Returns:
            Title with newline character if wrapping is needed
        """
        if len(title) <= max_chars:
            return title

        # Try to find good break points
        # Priority: 'vs', ' - ', ', ', ' '

        # First, try breaking at 'vs' if it exists and is roughly in the middle
        if ' vs ' in title:
            parts = title.split(' vs ')
            if len(parts) == 2:
                # Check if breaking here makes sense
                if len(parts[0]) <= max_chars and len(parts[1]) <= max_chars:
                    return f"{parts[0]} vs\n{parts[1]}"

        # Try breaking at ' - '
        if ' - ' in title:
            parts = title.split(' - ', 1)
            if len(parts[0]) <= max_chars:
                return f"{parts[0]} -\n{parts[1]}"

        # Try breaking at comma
        if ', ' in title:
            parts = title.split(', ', 1)
            if len(parts[0]) <= max_chars:
                return f"{parts[0]},\n{parts[1]}"

        # Last resort: break at space closest to middle
        words = title.split()
        mid_point = len(title) // 2
        best_break = 0
        min_distance = float('inf')

        current_pos = 0
        for i, word in enumerate(words[:-1]):  # Don't break after last word
            current_pos += len(word) + 1  # +1 for space
            distance = abs(current_pos - mid_point)
            if distance < min_distance:
                min_distance = distance
                best_break = i + 1

        if best_break > 0:
            first_line = ' '.join(words[:best_break])
            second_line = ' '.join(words[best_break:])
            return f"{first_line}\n{second_line}"

        # If all else fails, return original
        return title


    def anderson_normality(self,
                           data, alpha=0.05):
        """
        Perform the Anderson–Darling test for normality.

        Parameters:
            data (array-like): Input dataset.
            alpha (float): Significance level (default 0.05).

        Returns:
            dict: Contains statistic, critical value, and normality decision.
        """
        data = np.asarray(data)
        result = anderson(data, dist='norm')

        # Find the critical value corresponding to the chosen alpha
        # SciPy gives significance levels in percent: [15, 10, 5, 2.5, 1]
        idx = np.where(result.significance_level == (alpha * 100))[0]
        if len(idx) == 0:
            raise ValueError("Alpha must be one of: 0.15, 0.10, 0.05, 0.025, 0.01")

        critical_value = result.critical_values[idx[0]]
        statistic = result.statistic

        return {
            "statistic": statistic,
            "critical_value": critical_value,
            "alpha": alpha,
            "is_normal": statistic < critical_value
        }

    def plotting_box_plots(self,
                           parameters: list[str],
                           conventional_df: pd.DataFrame,
                           alternative_df: pd.DataFrame,
                           name: str = 'box_plot',
                           show_outliers: bool = True,
                           print_stats: bool = False,
                           alphas: list[float] = None,
                           use_different_colors: bool = True,
                           show_median_lines: bool = True,
                           show_mean_lines: bool = False,
                           shade_between: bool = True):  # New parameters

        conventional_fleet = conventional_df.copy()
        alternative_fleet = alternative_df.copy()

        # Validate alphas list
        if alphas is not None and len(alphas) != 2:
            self.logger.warning(
                f"Warning: alphas list length ({len(alphas)}) doesn't match number of datasets (2). Using default transparency.")
            alphas = None

        if alphas is None:
            alphas = [0.6, 0.8]

        # Set colors based on use_different_colors flag
        if use_different_colors:
            box_colors = ['blue', 'orange']  # Default distinct colors
        else:
            box_colors = ['blue', 'blue']  # Same color for both

        def get_stats(df: pd.Series, name: str):
            print(f'Stats for {name}: {df.describe()}')

        n_plots = len(parameters)
        cols = math.ceil(math.sqrt(n_plots))
        rows = math.ceil(n_plots / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(15, 12))

        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, param in enumerate(parameters):
            ax = axes[i]

            conventional_data = conventional_fleet[param].dropna()
            alternative_data = alternative_fleet[param].dropna()

            if len(conventional_data) == 0 and len(alternative_data) == 0:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
                ax.set_title(FleetPlottingClass.wrap_title(f'{param} - No Data'))
                continue

            plot_data = []
            labels = []
            box_alphas = []
            plot_colors = []
            positions = []  # Track box positions

            current_pos = 1
            if len(conventional_data) > 0:
                plot_data.append(conventional_data)
                labels.append(conventional_df)
                box_alphas.append(alphas[0])
                plot_colors.append(box_colors[0])
                positions.append(current_pos)
                current_pos += 1

            if len(alternative_data) > 0:
                plot_data.append(alternative_data)
                labels.append(alternative_df)
                box_alphas.append(alphas[1])
                plot_colors.append(box_colors[1])
                positions.append(current_pos)

            box_plot = ax.boxplot(plot_data,
                                  positions=positions,
                                  tick_labels=labels,
                                  showfliers=show_outliers,
                                  patch_artist=True)

            # Set transparency and color for each box
            for patch, alpha, color in zip(box_plot['boxes'], box_alphas, plot_colors):
                patch.set_alpha(alpha)
                patch.set_facecolor(color)

            # Calculate statistics for both datasets
            stats = {}
            for j, (data, label) in enumerate(zip(plot_data, labels)):
                stats[label] = {
                    'median': data.median(),
                    'mean': data.mean(),
                    'position': positions[j]
                }

            # Draw median/mean lines and shading if both datasets exist
            if len(plot_data) == 2:
                conv_label = labels[0]
                alt_label = labels[1]

                # Get x-axis limits for lines
                xlim = ax.get_xlim()

                # Draw median lines
                if show_median_lines:
                    conv_median = stats[conv_label]['median']
                    alt_median = stats[alt_label]['median']

                    # Draw horizontal dotted lines
                    ax.axhline(y=conv_median, color=plot_colors[0],
                               linestyle='--', linewidth=1.5, alpha=0.7,
                               label=f'{conv_label} median')
                    ax.axhline(y=alt_median, color=plot_colors[1],
                               linestyle='--', linewidth=1.5, alpha=0.7,
                               label=f'{alt_label} median')

                    # Shade between median lines
                    if shade_between:
                        # Determine if alternative is higher or lower
                        if alt_median > conv_median:
                            shade_color = 'green'
                            shade_alpha = 0.15
                        else:
                            shade_color = 'red'
                            shade_alpha = 0.15

                        ax.fill_between(xlim, conv_median, alt_median,
                                        color=shade_color, alpha=shade_alpha,
                                        zorder=0)  # Behind everything else

                # Draw mean lines
                if show_mean_lines:
                    conv_mean = stats[conv_label]['mean']
                    alt_mean = stats[alt_label]['mean']

                    # Draw horizontal dotted lines (different style from median)
                    ax.axhline(y=conv_mean, color=plot_colors[0],
                               linestyle=':', linewidth=2, alpha=0.7,
                               label=f'{conv_label} mean')
                    ax.axhline(y=alt_mean, color=plot_colors[1],
                               linestyle=':', linewidth=2, alpha=0.7,
                               label=f'{alt_label} mean')

                    # Optional: shade between mean lines too (if median not shown)
                    if shade_between and not show_median_lines:
                        if alt_mean > conv_mean:
                            shade_color = 'green'
                            shade_alpha = 0.15
                        else:
                            shade_color = 'red'
                            shade_alpha = 0.15

                        ax.fill_between(xlim, conv_mean, alt_mean,
                                        color=shade_color, alpha=shade_alpha,
                                        zorder=0)

                # Add small legend for the lines (positioned to not overlap with outlier box)
                if show_median_lines or show_mean_lines:
                    ax.legend(loc='lower right', fontsize=7, framealpha=0.8)

            ax.set_title(FleetPlottingClass.wrap_title(f'{param} Distribution'))

            # Calculate outlier statistics
            outlier_text_lines = []

            for j, (data, label) in enumerate(zip(plot_data, labels)):
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_fence = Q1 - 1.5 * IQR
                upper_fence = Q3 + 1.5 * IQR

                lower_outliers = data[data < lower_fence]
                upper_outliers = data[data > upper_fence]

                outlier_info = []
                if len(lower_outliers) > 0:
                    outlier_info.append(f'Low: {len(lower_outliers)}')
                if len(upper_outliers) > 0:
                    outlier_info.append(f'High: {len(upper_outliers)}')

                total_outliers = len(lower_outliers) + len(upper_outliers)
                total_count = len(data)
                pct = (total_outliers / total_count) * 100 if total_count > 0 else 0

                if total_outliers > 0:
                    detail_text = f' - {", ".join(outlier_info)}'
                else:
                    detail_text = ''

                outlier_text_lines.append(f'{label}: {total_outliers} outliers ({pct:.1f}%){detail_text}')

            outlier_text = '\n'.join(outlier_text_lines)
            visibility_note = "" if show_outliers else " (hidden)"
            full_text = f"Outliers{visibility_note}:\n{outlier_text}"

            ax.text(0.02, 0.98, full_text, transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=8)

            if print_stats:
                if len(conventional_data) > 0:
                    get_stats(conventional_data, f'{param} for conventional fuel')
                if len(alternative_data) > 0:
                    get_stats(alternative_data, f'{param} for alternative fuel')

                self.logger.info(f'Plot made for {param} (Conv: {len(conventional_data)}, Alt: {len(alternative_data)})')

        for i in range(n_plots, rows * cols):
            axes[i].axis('off')

        fig.suptitle(f'{name} Distribution')
        plt.tight_layout()
        plt.show()

    def making_scatter_plots(self,
                             dataframes: dict[str, pd.DataFrame],
                             parameters: list[str],
                             x_axis: str | list[str],
                             show_outliers: bool = True,
                             name: str = '',
                             print_stats: bool = False,
                             colors: list[str] = None,
                             alphas: list[float] = None,
                             use_different_colors: bool = True):  # Add color toggle
        """
        Create scatter plots comparing multiple ship datasets.

        Args:
            dataframes: Dictionary containing pandas dataframes to be plotted (with their names)
            parameters: List of y-axis parameter names to plot
            x_axis: Single x-axis parameter name or list of x-axis parameters
            show_outliers: Whether to include outliers in plots
            name: Name for the overall plot title
            print_stats: Whether to print statistical summaries to console
            colors: Optional list of colors for each dataset (auto-generated if None)
            alphas: Optional list of transparency values (0-1) for each dataset
            use_different_colors: Whether to use different colors for each dataset (default True)
        """

        n_datasets = len(dataframes)

        # Handle colors based on use_different_colors flag
        if use_different_colors:
            if colors is None:
                import matplotlib.cm as cm
                colors = [cm.tab10(i % 10) for i in range(n_datasets)]
        else:
            colors = ['blue'] * n_datasets  # All same color

        # Validate and set alphas
        if alphas is not None and len(alphas) != n_datasets:
            self.logger.warning(
                f"Warning: alphas list length ({len(alphas)}) doesn't match number of datasets ({n_datasets}). Using default transparency.")
            alphas = None

        if alphas is None:
            alphas = [0.6] * n_datasets

        x_axes = [x_axis] if isinstance(x_axis, str) else x_axis
        n_plots = len(parameters)
        cols = math.ceil(math.sqrt(n_plots))
        rows = math.ceil(n_plots / cols)

        def get_stats(df: pd.Series, name: str):
            if print_stats:
                print(f'Stats for {name}: {df.describe()}')

        for x_param in x_axes:
            print(f"\n=== Creating plots with {x_param} as X-axis ===")

            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6 * rows))

            if n_plots == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for i, y_param in enumerate(parameters):
                ax = axes[i]
                has_data = False

                df_names = dataframes.keys()

                for j, df_name in enumerate(df_names):
                    df = dataframes[df_name]
                    df_clean = df[[x_param, y_param]].dropna()

                    if len(df_clean) > 0:
                        has_data = True
                        ax.scatter(df_clean[x_param], df_clean[y_param],
                                   alpha=alphas[j],
                                   label=df_name,
                                   color=colors[j],
                                   s=20)

                        if print_stats:
                            get_stats(df_clean[y_param], f'{y_param} for {df_name}')
                            print(f'  {df_name}: {len(df_clean)} points')

                if not has_data:
                    ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
                    ax.set_title(FleetPlottingClass.wrap_title(f'{y_param} vs {x_param} - No Data'))
                    continue

                ax.set_xlabel(x_param)
                ax.set_ylabel(y_param)
                ax.set_title(FleetPlottingClass.wrap_title(f'{y_param} vs {x_param}'))
                ax.legend()
                ax.grid(True, alpha=0.3)

            for i in range(n_plots, rows * cols):
                axes[i].axis('off')

            fig.suptitle(f'{name} Parameter Relationships with {x_param}', fontsize=16, y=0.98)
            plt.tight_layout()
            plt.show()

    def making_3d_scatter_plots(self,
                                dataframes: dict[str,pd.DataFrame],
                                parameters: list[tuple[str, str, str]],
                                show_outliers: bool = True,
                                name: str = '',
                                print_stats: bool = True,
                                colors: list[str] = None,
                                alphas: list[float] = None,
                                use_different_colors: bool = True):  # Add color toggle
        """
        Create 3D scatter plots comparing multiple ship datasets.

        Args:
            dataframes: Dictionary containing pandas dataframes to be plotted (with their names)
            parameters: List of tuples (x_param, y_param, z_param) for each 3D plot
            show_outliers: Whether to include outliers in plots
            name: Name for the overall plot title
            print_stats: Whether to print statistical summaries to console
            colors: Optional list of colors for each dataset (auto-generated if None)
            alphas: Optional list of transparency values (0-1) for each dataset
            use_different_colors: Whether to use different colors for each dataset (default True)
        """

        n_datasets = len(dataframes)

        # Handle colors based on use_different_colors flag
        if use_different_colors:
            if colors is None:
                import matplotlib.cm as cm
                colors = [cm.tab10(i % 10) for i in range(n_datasets)]
        else:
            colors = ['blue'] * n_datasets  # All same color

        # Validate and set alphas
        if alphas is not None and len(alphas) != n_datasets:
            self.logger.warning(
                f"Warning: alphas list length ({len(alphas)}) doesn't match number of datasets ({n_datasets}). Using default transparency.")
            alphas = None

        if alphas is None:
            alphas = [0.6] * n_datasets

        def get_stats(df: pd.Series, name: str):
            if print_stats:
                print(f'Stats for {name}: {df.describe()}')

        n_plots = len(parameters)
        cols = math.ceil(math.sqrt(n_plots))
        rows = math.ceil(n_plots / cols)

        fig = plt.figure(figsize=(7 * cols, 7 * rows))

        for i, (x_param, y_param, z_param) in enumerate(parameters):
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')

            has_data = False

            df_names = dataframes.keys()

            for j, df_name in enumerate(df_names):
                df = dataframes[df_name]
                df_clean = df[[x_param, y_param, z_param]].dropna()

                if len(df_clean) > 0:
                    has_data = True
                    ax.scatter(df_clean[x_param], df_clean[y_param], df_clean[z_param],
                               alpha=alphas[j],
                               label=df_name,
                               color=colors[j],
                               s=20)

                    if print_stats:
                        get_stats(df_clean[z_param], f'{z_param} for {df_name}')
                        print(f'  {df_name}: {len(df_clean)} points')

            if not has_data:
                ax.text2D(0.5, 0.5, 'No data available',
                          ha='center', va='center', transform=ax.transAxes)
                ax.set_title(FleetPlottingClass.wrap_title(f'{z_param} vs {y_param} vs {x_param} - No Data'))
                continue

            ax.set_xlabel(x_param)
            ax.set_ylabel(y_param)
            ax.set_zlabel(z_param)
            ax.set_title(FleetPlottingClass.wrap_title(f'{z_param} vs {y_param} vs {x_param}'))
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'{name} 3D Parameter Relationships', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()

    def making_parallel_coordinates(self,
                                    dataframes: dict[str,pd.DataFrame],
                                    parameters: list[str],
                                    name: str = 'Parallel Coordinates',
                                    colors: list[str] = None,
                                    alphas: list[float] = None,
                                    use_different_colors: bool = True,
                                    normalize: bool = True,
                                    print_stats: bool = True):
        """
        Create parallel coordinate plots comparing multiple ship datasets.

        Args:
            dataframes: Dictionary containing pandas dataframes to be plotted (with their names)
            parameters: List of parameters to include as axes in the plot
            name: Title for the plot
            colors: Optional list of colors for each dataset
            alphas: Optional list of transparency values (0-1) for each dataset
            use_different_colors: Whether to use different colors for each dataset (default True)
            normalize: Whether to normalize data to 0-1 scale for better visualization (default True)
            print_stats: Whether to print statistics to console
        """

        n_datasets = len(dataframes)

        # Handle colors
        if use_different_colors:
            if colors is None:
                import matplotlib.cm as cm
                colors = [cm.tab10(i % 10) for i in range(n_datasets)]
        else:
            colors = ['blue'] * n_datasets

        # Validate and set alphas
        if alphas is not None and len(alphas) != n_datasets:
            self.logger.warning(
                f"Warning: alphas list length ({len(alphas)}) doesn't match number of datasets ({n_datasets}). Using default transparency.")
            alphas = None

        if alphas is None:
            alphas = [0.3] * n_datasets  # Lower default alpha for parallel coords

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Determine global min/max for normalization if needed
        if normalize:
            global_mins = {}
            global_maxs = {}

            for param in parameters:
                all_values = []

                for df_name,df in dataframes.items():
                    if param in df.columns:
                        all_values.extend(df[param].dropna().values)

                if all_values:
                    global_mins[param] = min(all_values)
                    global_maxs[param] = max(all_values)
                else:
                    global_mins[param] = 0
                    global_maxs[param] = 1

        # Plot each dataset
        df_names = dataframes.keys()

        for idx, df_name in enumerate(df_names):
            df = dataframes[df_name]

            # Select only the parameters that exist in the dataframe
            available_params = [p for p in parameters if p in df.columns]

            if not available_params:
                self.logger.warning(f"Warning: No parameters found in {df_name}")
                continue

            # Get clean data (drop rows with any NaN in selected parameters)
            plot_data = df[available_params].dropna()

            if len(plot_data) == 0:
                self.logger.warning(f"Warning: No complete data rows for {df_name}")
                continue

            if print_stats:
                print(f"\n{df_name}: {len(plot_data)} complete rows")
                print(f"Parameters: {available_params}")

            # Normalize data if requested
            if normalize:
                normalized_data = plot_data.copy()
                for param in available_params:
                    min_val = global_mins[param]
                    max_val = global_maxs[param]
                    if max_val - min_val > 0:
                        normalized_data[param] = (plot_data[param] - min_val) / (max_val - min_val)
                    else:
                        normalized_data[param] = 0.5  # If all values same, put in middle
                plot_data = normalized_data

            # Plot each row as a line
            for _, row in plot_data.iterrows():
                ax.plot(range(len(available_params)), row.values,
                        color=colors[idx], alpha=alphas[idx], linewidth=0.5)

        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=colors[i], linewidth=2,
                                  label=df_names[i])
                           for i in range(n_datasets)]
        ax.legend(handles=legend_elements, loc='upper right')

        # Set up axes
        ax.set_xticks(range(len(parameters)))
        ax.set_xticklabels([FleetPlottingClass.wrap_title(p, max_chars=15) for p in parameters], rotation=45, ha='right')
        ax.set_xlim(-0.5, len(parameters) - 0.5)

        if normalize:
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel('Normalized Value (0-1)')

            # Add actual value ranges as text
            for i, param in enumerate(parameters):
                if param in global_mins and param in global_maxs:
                    min_val = global_mins[param]
                    max_val = global_maxs[param]
                    # Add small text showing actual range
                    ax.text(i, -0.1, f'[{min_val:.1f}-{max_val:.1f}]',
                            ha='center', fontsize=7, rotation=45)
        else:
            ax.set_ylabel('Value')

        ax.set_xlabel('Parameters')
        ax.set_title(f'{name} - Parallel Coordinates Plot')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

    def making_parallel_coordinates_separate(self,
                                                dataframes: dict[str,pd.DataFrame],
                                             parameters: list[str],
                                             name: str = 'Parallel Coordinates',
                                             colors: list[str] = None,
                                             alphas: list[float] = None,
                                             use_different_colors: bool = True,
                                             normalize: bool = True,
                                             print_stats: bool = True):
        """
        Create separate parallel coordinate plots for each dataset (one subplot per dataset).

        Args:
            dataframes: Dictionary containing pandas dataframes to be plotted (with their names)
            parameters: List of parameters to include as axes in the plot
            name: Title for the overall figure
            colors: Optional list of colors for each dataset
            alphas: Optional list of transparency values (0-1) for each dataset
            use_different_colors: Whether to use different colors for each dataset (default True)
            normalize: Whether to normalize data to 0-1 scale for better visualization (default True)
            print_stats: Whether to print statistics to console
        """

        n_datasets = len(dataframes)

        # Handle colors
        if use_different_colors:
            if colors is None:
                import matplotlib.cm as cm
                colors = [cm.tab10(i % 10) for i in range(n_datasets)]
        else:
            colors = ['blue'] * n_datasets

        # Validate and set alphas
        if alphas is not None and len(alphas) != n_datasets:
            self.logger.warning(
                f"Warning: alphas list length ({len(alphas)}) doesn't match number of datasets ({n_datasets}). Using default transparency.")
            alphas = None

        if alphas is None:
            alphas = [0.3] * n_datasets

        # Calculate layout
        n_plots = n_datasets
        cols = min(2, n_plots)  # Max 2 columns
        rows = math.ceil(n_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(12 * cols, 6 * rows))

        # Handle single subplot case
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Determine global min/max for normalization if needed
        if normalize:
            global_mins = {}
            global_maxs = {}

            for param in parameters:
                all_values = []
                for df_name,df in dataframes.items():
                    if param in df.columns:
                        all_values.extend(df[param].dropna().values)

                if all_values:
                    global_mins[param] = min(all_values)
                    global_maxs[param] = max(all_values)
                else:
                    global_mins[param] = 0
                    global_maxs[param] = 1

        # Plot each dataset in its own subplot
        df_names = dataframes.keys()
        for idx, df_name in enumerate(df_names):
            ax = axes[idx]
            df = dataframes[df_name]

            # Select only the parameters that exist in the dataframe
            available_params = [p for p in parameters if p in df.columns]

            if not available_params:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{df_name} - No Data')
                continue

            # Get clean data
            plot_data = df[available_params].dropna()

            if len(plot_data) == 0:
                ax.text(0.5, 0.5, 'No complete rows', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{df_name} - No Data')
                continue

            if print_stats:
                print(f"\n{df_name}: {len(plot_data)} complete rows")

            # Normalize data if requested
            if normalize:
                normalized_data = plot_data.copy()
                for param in available_params:
                    min_val = global_mins[param]
                    max_val = global_maxs[param]
                    if max_val - min_val > 0:
                        normalized_data[param] = (plot_data[param] - min_val) / (max_val - min_val)
                    else:
                        normalized_data[param] = 0.5
                plot_data = normalized_data

            # Plot each row as a line
            for _, row in plot_data.iterrows():
                ax.plot(range(len(available_params)), row.values,
                        color=colors[idx], alpha=alphas[idx], linewidth=0.5)

            # Set up axes
            ax.set_xticks(range(len(available_params)))
            ax.set_xticklabels([FleetPlottingClass.wrap_title(p, max_chars=15) for p in available_params],
                               rotation=45, ha='right')
            ax.set_xlim(-0.5, len(available_params) - 0.5)

            if normalize:
                ax.set_ylim(-0.05, 1.05)
                ax.set_ylabel('Normalized Value (0-1)')
            else:
                ax.set_ylabel('Value')

            ax.set_xlabel('Parameters')
            ax.set_title(f'{df_name} ({len(plot_data)} ships)')
            ax.grid(True, alpha=0.3, axis='y')

        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')

        fig.suptitle(f'{name} - Parallel Coordinates Plots', fontsize=16, y=0.99)
        plt.tight_layout()
        plt.show()

    def split_and_plot(self,
                       dataframes: dict[str,pd.DataFrame],
                       plot_function: callable,
                       split_column: str,
                       n_bins: int,
                       bin_labels: list[str] = None,
                       **plot_kwargs):
        """
        Wrapper function that splits data into bins and creates subplots for each bin.

        :param dataframes: Dictionary containing pandas dataframes to be plotted (with their names)
        :param plot_function: The plotting function to use (making_scatter_plots, making_parallel_coordinates, etc.)
        :param split_column: Column name to split data by
        :param n_bins: Number of bins to split data into
        :param bin_labels: Optional custom labels for bins (e.g., ['Small', 'Medium', 'Large'])
        :param plot_kwargs: All other arguments to pass to the plotting function
        """

        # Collect all data for the split column to determine bin edges
        all_split_data = []
        for df_name in dataframes.keys():
            df = dataframes[df_name]
            if split_column in df.columns:
                all_split_data.extend(df[split_column].dropna().values)

        if not all_split_data:
            self.logger.warning(f"Error: No data found in column '{split_column}'")
            return

        # Create bins using equal-width for comparing ships of similar sizes
        bin_edges = np.linspace(min(all_split_data), max(all_split_data), n_bins + 1)

        # Ensure unique bin edges (in case of repeated values)
        bin_edges = np.unique(bin_edges)
        actual_n_bins = len(bin_edges) - 1

        if actual_n_bins < n_bins:
            self.logger.warning(f"Warning: Only {actual_n_bins} unique bins possible (requested {n_bins})")

        # Generate bin labels if not provided
        if bin_labels is None:
            bin_labels = [f'{split_column} [{bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f}]'
                          for i in range(actual_n_bins)]

        print(f"\nSplitting data by {split_column} into {actual_n_bins} bins:")
        for i, label in enumerate(bin_labels):
            print(f"  Bin {i + 1}: {label}")

        # IMPORTANT: Save the original name BEFORE the loop
        original_name = plot_kwargs.get('name', '')

        # Create a temporary FleetData object for each bin
        for bin_idx in range(actual_n_bins):
            print(f"\n{'=' * 60}")
            print(f"Processing Bin {bin_idx + 1}/{actual_n_bins}: {bin_labels[bin_idx]}")
            print(f"{'=' * 60}")

            # Create filtered dataframes for this bin
            from fleet_data_class import FleetData
            bin_fleet = FleetData()  # Create new FleetData instance
            bin_fleet.processed_data = {}

            lower_bound = bin_edges[bin_idx]
            upper_bound = bin_edges[bin_idx + 1]

            # Include upper bound for last bin
            if bin_idx == actual_n_bins - 1:
                mask_func = lambda x: (x >= lower_bound) & (x <= upper_bound)
            else:
                mask_func = lambda x: (x >= lower_bound) & (x < upper_bound)

            bin_empty = True
            for df_name,df in dataframes.items():

                if split_column in df.columns:
                    # Filter data for this bin
                    mask = mask_func(df[split_column])
                    bin_df = df[mask].copy()

                    if len(bin_df) > 0:
                        bin_fleet.processed_data[df_name] = bin_df
                        bin_empty = False
                        print(f"  {df_name}: {len(bin_df)} ships")
                    else:
                        # Add empty dataframe to maintain structure
                        bin_fleet.processed_data[df_name] = df.iloc[0:0].copy()
                        print(f"  {df_name}: 0 ships")
                else:
                    # If split column doesn't exist, include all data
                    bin_fleet.processed_data[df_name] = df.copy()

            if bin_empty:
                print(f"  Skipping empty bin")
                continue

            # Update the plot name to include bin info using the SAVED original_name
            plot_kwargs['name'] = f"{original_name} - {bin_labels[bin_idx]}"

            # Call the plotting function with the binned data
            plot_function(fleet=bin_fleet, **plot_kwargs)

    def plotting_histograms(self,
                            parameters: list[str],
                            conventional_df: pd.DataFrame,
                            alternative_df: pd.DataFrame,
                            name: str = 'histogram',
                            bins: int = 30,
                            print_stats: bool = False,
                            alphas: list[float] = None,
                            colors: list[str] = None,
                            use_different_colors: bool = True,
                            density: bool = False,
                            show_kde: bool = True,
                            show_mean: bool = True,
                            show_median: bool = True):
        """
        Create histogram plots comparing conventional and alternative fuel ship datasets.

        The histogram shows the frequency distribution of values, helping identify:
        - Distribution shape (normal, skewed, bimodal)
        - Central tendency and spread
        - Outliers and gaps in data

        Args:
            parameters: List of parameter names to plot
            conventional_df: DataFrame containing conventional fuel ships
            alternative_df: DataFrame containing alternative fuel ships
            name: Name for the overall plot title
            bins: Number of bins for histogram (default 30)
            print_stats: Whether to print statistical summaries
            alphas: List of transparency values for [conventional, alternative]
            colors: List of colors for [conventional, alternative]
            use_different_colors: Whether to use different colors for each dataset
            density: If True, plot probability density instead of counts
            show_kde: Whether to overlay kernel density estimation curve
            show_mean: Whether to show vertical line at mean
            show_median: Whether to show vertical line at median
        """

        conventional_fleet = conventional_df.copy()
        alternative_fleet = alternative_df.copy()

        # Validate alphas
        if alphas is not None and len(alphas) != 2:
            self.logger.warning(
                f"Warning: alphas list length ({len(alphas)}) doesn't match number of datasets (2). Using default.")
            alphas = None

        if alphas is None:
            alphas = [0.6, 0.6]

        # Set colors
        if use_different_colors:
            if colors is None:
                colors = ['blue', 'orange']
        else:
            colors = ['blue', 'blue']

        n_plots = len(parameters)
        cols = math.ceil(math.sqrt(n_plots))
        rows = math.ceil(n_plots / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(15, 12))

        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, param in enumerate(parameters):
            ax = axes[i]

            conventional_data = conventional_fleet[param].dropna()
            alternative_data = alternative_fleet[param].dropna()

            if len(conventional_data) == 0 and len(alternative_data) == 0:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
                ax.set_title(FleetPlottingClass.wrap_title(f'{param} - No Data'))
                continue

            # Determine common bin edges for fair comparison
            all_data = []
            if len(conventional_data) > 0:
                all_data.extend(conventional_data)
            if len(alternative_data) > 0:
                all_data.extend(alternative_data)

            data_min = min(all_data)
            data_max = max(all_data)
            bin_edges = np.linspace(data_min, data_max, bins + 1)

            # Plot histograms
            if len(conventional_data) > 0:
                ax.hist(conventional_data, bins=bin_edges, alpha=alphas[0],
                        color=colors[0], label='Conventional',
                        density=density, edgecolor='black', linewidth=0.5)

                # Add mean and median lines
                if show_mean:
                    mean_val = conventional_data.mean()
                    ax.axvline(mean_val, color=colors[0], linestyle='--',
                               linewidth=2, alpha=0.8, label=f'Conv. Mean: {mean_val:.2f}')

                if show_median:
                    median_val = conventional_data.median()
                    ax.axvline(median_val, color=colors[0], linestyle=':',
                               linewidth=2, alpha=0.8, label=f'Conv. Median: {median_val:.2f}')

            if len(alternative_data) > 0:
                ax.hist(alternative_data, bins=bin_edges, alpha=alphas[1],
                        color=colors[1], label='Alternative',
                        density=density, edgecolor='black', linewidth=0.5)

                # Add mean and median lines
                if show_mean:
                    mean_val = alternative_data.mean()
                    ax.axvline(mean_val, color=colors[1], linestyle='--',
                               linewidth=2, alpha=0.8, label=f'Alt. Mean: {mean_val:.2f}')

                if show_median:
                    median_val = alternative_data.median()
                    ax.axvline(median_val, color=colors[1], linestyle=':',
                               linewidth=2, alpha=0.8, label=f'Alt. Median: {median_val:.2f}')

            # Add KDE curves if requested
            if show_kde:
                from scipy import stats

                if len(conventional_data) > 1:
                    kde = stats.gaussian_kde(conventional_data)
                    x_range = np.linspace(data_min, data_max, 200)
                    kde_values = kde(x_range)

                    # Scale KDE to match histogram
                    if not density:
                        kde_values = kde_values * len(conventional_data) * (data_max - data_min) / bins

                    ax.plot(x_range, kde_values, color=colors[0],
                            linewidth=2, alpha=0.8, label='Conv. KDE')

                if len(alternative_data) > 1:
                    kde = stats.gaussian_kde(alternative_data)
                    x_range = np.linspace(data_min, data_max, 200)
                    kde_values = kde(x_range)

                    if not density:
                        kde_values = kde_values * len(alternative_data) * (data_max - data_min) / bins

                    ax.plot(x_range, kde_values, color=colors[1],
                            linewidth=2, alpha=0.8, label='Alt. KDE')

            ax.set_xlabel(param)
            ax.set_ylabel('Density' if density else 'Frequency')
            ax.set_title(FleetPlottingClass.wrap_title(f'{param} Distribution'))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

            if print_stats:
                if len(conventional_data) > 0:
                    self.logger.info(f'\nConventional {param}:')
                    self.logger.info(f'  Count: {len(conventional_data)}')
                    self.logger.info(f'  Mean: {conventional_data.mean():.2f}')
                    self.logger.info(f'  Median: {conventional_data.median():.2f}')
                    self.logger.info(f'  Std: {conventional_data.std():.2f}')

                if len(alternative_data) > 0:
                    self.logger.info(f'\nAlternative {param}:')
                    self.logger.info(f'  Count: {len(alternative_data)}')
                    self.logger.info(f'  Mean: {alternative_data.mean():.2f}')
                    self.logger.info(f'  Median: {alternative_data.median():.2f}')
                    self.logger.info(f'  Std: {alternative_data.std():.2f}')

        # Hide unused subplots
        for i in range(n_plots, rows * cols):
            axes[i].axis('off')

        fig.suptitle(f'{name} Histogram Comparison')
        plt.tight_layout()
        plt.show()

    def plotting_qq_plots(self,
                          parameters: list[str],
                          conventional_df: pd.DataFrame,
                          alternative_df: pd.DataFrame,
                          name: str = 'Q-Q Plot',
                          print_stats: bool = False,
                          colors: list[str]|None = None,
                          use_different_colors: bool = True,
                          comparison_type: str = 'theoretical',
                          show_reference_line: bool = True,
                          show_confidence_bands: bool = True,
                          use_log_transform: bool = False):
        """
        Create Q-Q (Quantile-Quantile) plots for distribution comparison.

        Q-Q plots compare quantiles of your data against:
        - 'theoretical': Normal distribution (tests normality assumption)
        - 'datasets': Each other (compares if both follow same distribution)

        Points along the diagonal line indicate distributions match.
        Deviations show where and how distributions differ.

        Args:
            parameters: List of parameter names to plot
            conventional_df: DataFrame containing conventional fuel ships
            alternative_df: DataFrame containing alternative fuel ships
            name: Name for the overall plot title
            print_stats: Whether to print statistical summaries
            colors: List of colors for [conventional, alternative]
            use_different_colors: Whether to use different colors for each dataset
            comparison_type: 'theoretical' (vs normal dist) or 'datasets' (vs each other)
            show_reference_line: Whether to show the theoretical diagonal line
            show_confidence_bands: Whether to show confidence bands (theoretical only)
            use_log_transform: If True, apply log transformation before plotting.
                              Useful for right-skewed data like DWT, power, etc.
                              Note: Negative and zero values will be excluded.
        """

        from scipy import stats

        conventional_fleet = conventional_df.copy()
        alternative_fleet = alternative_df.copy()

        # Set colors
        if use_different_colors:
            if colors is None:
                colors = ['blue', 'orange']
        else:
            colors = ['blue', 'blue']

        n_plots = len(parameters)
        cols = math.ceil(math.sqrt(n_plots))
        rows = math.ceil(n_plots / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(15, 12))

        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, param in enumerate(parameters):
            ax = axes[i]

            conventional_data = conventional_fleet[param].dropna()
            alternative_data = alternative_fleet[param].dropna()

            # Apply log transformation if requested
            if use_log_transform:
                # Filter out non-positive values before log transform
                conv_positive = conventional_data > 0
                alt_positive = alternative_data > 0

                n_conv_removed = len(conventional_data) - conv_positive.sum()
                n_alt_removed = len(alternative_data) - alt_positive.sum()

                if n_conv_removed > 0:
                    self.logger.warning(
                        f"Warning: Removed {n_conv_removed} non-positive values from conventional {param} before log transform")
                if n_alt_removed > 0:
                    self.logger.warning(
                        f"Warning: Removed {n_alt_removed} non-positive values from alternative {param} before log transform")

                conventional_data = np.log(conventional_data[conv_positive])
                alternative_data = np.log(alternative_data[alt_positive])

            if len(conventional_data) == 0 and len(alternative_data) == 0:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
                title_suffix = ' - No Data'
                if use_log_transform:
                    title_suffix = ' (log) - No Data'
                ax.set_title(FleetPlottingClass.wrap_title(f'{param}{title_suffix}'))
                continue

            if comparison_type == 'theoretical':
                # Compare each dataset against theoretical normal distribution

                if len(conventional_data) > 0:
                    # Generate theoretical quantiles and sample quantiles
                    (osm, osr), (slope, intercept, r) = stats.probplot(
                        conventional_data, dist="norm", plot=None)

                    ax.scatter(osm, osr, alpha=0.6, color=colors[0],
                               s=30, label=f'Conventional (R²={r ** 2:.3f})', zorder=3)

                    if show_reference_line:
                        # Theoretical line (perfect normal distribution)
                        ax.plot(osm, slope * osm + intercept, 'k--',
                                linewidth=1.5, alpha=0.7, label='Theoretical Normal', zorder=2)

                    if show_confidence_bands:
                        # Add 95% confidence bands
                        # These show expected variation if data were truly normal
                        std_err = np.std(osr - (slope * osm + intercept))
                        ax.fill_between(osm,
                                        slope * osm + intercept - 1.96 * std_err,
                                        slope * osm + intercept + 1.96 * std_err,
                                        alpha=0.2, color=colors[0], zorder=1)

                if len(alternative_data) > 0:
                    (osm, osr), (slope, intercept, r) = stats.probplot(
                        alternative_data, dist="norm", plot=None)

                    ax.scatter(osm, osr, alpha=0.6, color=colors[1],
                               s=30, label=f'Alternative (R²={r ** 2:.3f})', zorder=3)

                    if show_reference_line and len(conventional_data) == 0:
                        ax.plot(osm, slope * osm + intercept, 'k--',
                                linewidth=1.5, alpha=0.7, label='Theoretical Normal', zorder=2)

                    if show_confidence_bands:
                        std_err = np.std(osr - (slope * osm + intercept))
                        ax.fill_between(osm,
                                        slope * osm + intercept - 1.96 * std_err,
                                        slope * osm + intercept + 1.96 * std_err,
                                        alpha=0.2, color=colors[1], zorder=1)

                ax.set_xlabel('Theoretical Quantiles (Normal Distribution)')
                ylabel = 'Sample Quantiles (log scale)' if use_log_transform else 'Sample Quantiles'
                ax.set_ylabel(ylabel)

                param_label = f'log({param})' if use_log_transform else param
                ax.set_title(FleetPlottingClass.wrap_title(
                    f'{param_label} - Q-Q Plot vs Normal Distribution'))

            elif comparison_type == 'datasets':
                # Compare the two datasets against each other

                if len(conventional_data) > 0 and len(alternative_data) > 0:
                    # Get quantiles for both datasets
                    n_quantiles = min(len(conventional_data), len(alternative_data), 100)
                    quantile_levels = np.linspace(0, 1, n_quantiles)

                    conv_quantiles = np.quantile(conventional_data, quantile_levels)
                    alt_quantiles = np.quantile(alternative_data, quantile_levels)

                    ax.scatter(conv_quantiles, alt_quantiles, alpha=0.6,
                               color='purple', s=30, zorder=3)

                    if show_reference_line:
                        # Perfect match line (y=x)
                        min_val = min(conv_quantiles.min(), alt_quantiles.min())
                        max_val = max(conv_quantiles.max(), alt_quantiles.max())
                        ax.plot([min_val, max_val], [min_val, max_val],
                                'k--', linewidth=1.5, alpha=0.7,
                                label='Perfect Match (y=x)', zorder=2)

                    xlabel = 'Conventional Quantiles (log scale)' if use_log_transform else 'Conventional Quantiles'
                    ylabel = 'Alternative Quantiles (log scale)' if use_log_transform else 'Alternative Quantiles'
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)

                    param_label = f'log({param})' if use_log_transform else param
                    ax.set_title(FleetPlottingClass.wrap_title(
                        f'{param_label} - Q-Q Plot: Conv vs Alt'))

                    # Make axes equal for easier interpretation
                    ax.set_aspect('equal', adjustable='box')
                else:
                    ax.text(0.5, 0.5, 'Need both datasets for comparison',
                            ha='center', va='center')
                    param_label = f'log({param})' if use_log_transform else param
                    ax.set_title(FleetPlottingClass.wrap_title(f'{param_label} - Insufficient Data'))
                    continue

            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)

            if print_stats:
                param_label = f'log({param})' if use_log_transform else param

                if len(conventional_data) > 0:
                    # Shapiro-Wilk test for normality
                    if len(conventional_data) >= 3:
                        shapiro_stat, shapiro_p = stats.shapiro(conventional_data)
                        self.logger.info(f'\nConventional {param_label} - Normality Test:')
                        self.logger.info(f'  Shapiro-Wilk statistic: {shapiro_stat:.4f}')
                        self.logger.info(f'  Shapiro-Wilk p-value: {shapiro_p:.4f}')
                        self.logger.info(
                            f'  Interpretation: {"Likely normal" if shapiro_p > 0.05 else "Likely not normal"} (α=0.05)')

                        if use_log_transform:
                            self.logger.info(f'  Note: Statistics are for log-transformed data')

                if len(alternative_data) > 0:
                    if len(alternative_data) >= 3:
                        shapiro_stat, shapiro_p = stats.shapiro(alternative_data)
                        self.logger.info(f'\nAlternative {param_label} - Normality Test:')
                        self.logger.info(f'  Shapiro-Wilk statistic: {shapiro_stat:.4f}')
                        self.logger.info(f'  Shapiro-Wilk p-value: {shapiro_p:.4f}')
                        self.logger.info(
                            f'  Interpretation: {"Likely normal" if shapiro_p > 0.05 else "Likely not normal"} (α=0.05)')

                        if use_log_transform:
                            self.logger.info(f'  Note: Statistics are for log-transformed data')

                if comparison_type == 'datasets' and len(conventional_data) > 0 and len(alternative_data) > 0:
                    # Kolmogorov-Smirnov test to compare distributions
                    ks_stat, ks_p = stats.ks_2samp(conventional_data, alternative_data)
                    self.logger.info(f'\nDataset Comparison - {param_label}:')
                    self.logger.info(f'  Kolmogorov-Smirnov statistic: {ks_stat:.4f}')
                    self.logger.info(f'  Kolmogorov-Smirnov p-value: {ks_p:.4f}')
                    self.logger.info(
                        f'  Interpretation: {"Same distribution" if ks_p > 0.05 else "Different distributions"} (α=0.05)')

                    if use_log_transform:
                        self.logger.info(f'  Note: Comparison is on log-transformed data')

        # Hide unused subplots
        for i in range(n_plots, rows * cols):
            axes[i].axis('off')

        comparison_label = 'vs Normal Distribution' if comparison_type == 'theoretical' else 'Conv vs Alt'
        transform_label = ' (Log-Transformed)' if use_log_transform else ''
        fig.suptitle(f'{name} Q-Q Plot Comparison ({comparison_label}){transform_label}')
        plt.tight_layout()
        plt.show()

    def plotting_conditional_histograms(self,
                                        dataframes: dict[str, pd.DataFrame],
                                        x_variables: str|list[str],
                                        y_variables: list[str],
                                        name: str = 'Conditional Histogram',
                                        n_x_bins: int = 4,
                                        n_hist_bins: int = 20,
                                        x_bin_labels: list[str]|None = None,
                                        colors: list[str]|None = None,
                                        alphas: list[str]|None = None,
                                        use_different_colors: bool = True,
                                        density: bool = True,
                                        show_kde: bool = True,
                                        show_statistics: bool = True,
                                        print_stats: bool = False,
                                        stacked: bool = False,
                                        overlay_style: str = 'side_by_side'):
        """
        Create conditional histograms showing how y-variable distributions change across x-variable ranges.

        CONCEPT: Conditional histograms reveal how one variable's distribution (y) changes
        depending on the value of another variable (x). For example:
        - How does BEAM distribution differ for small vs large ships (binned by DWT)?
        - Does MCR distribution vary across different length categories (binned by LBP)?

        The x-variable is divided into bins (ranges), and for each bin we plot the
        distribution of the y-variable. This creates a matrix of histograms that shows
        relationships between variables.

        Args:
            dataframes: Dictionary of {dataset_name: DataFrame}
            x_variables: Variable(s) to bin and condition on (creates separate figure per x)
            y_variables: Variables to show distribution of (one subplot per y)
            name: Base name for plot titles
            n_x_bins: Number of bins to split x-variable into (e.g., 4 = quartiles)
            n_hist_bins: Number of bins within each histogram
            x_bin_labels: Custom labels for x bins (e.g., ['Small', 'Medium', 'Large', 'X-Large'])
            colors: List of colors for each dataset
            alphas: List of transparency values for each dataset
            use_different_colors: Whether to use different colors per dataset
            density: If True, normalize histograms to show probability density
            show_kde: Whether to overlay kernel density estimation curves
            show_statistics: Whether to display mean/median markers
            print_stats: Whether to print statistical summaries to console
            stacked: If True, stack histograms instead of overlaying
            overlay_style: 'side_by_side' (bars adjacent) or 'overlay' (bars overlapping)
        """

        # Convert single x_variable to list
        x_vars = [x_variables] if isinstance(x_variables, str) else x_variables

        n_datasets = len(dataframes)
        dataset_names = list(dataframes.keys())

        # Setup colors
        if use_different_colors:
            if colors is None:
                import matplotlib.cm as cm
                colors = [cm.tab10(i % 10) for i in range(n_datasets)]
        else:
            colors = ['blue'] * n_datasets

        # Setup alphas
        if alphas is not None and len(alphas) != n_datasets:
            self.logger.warning(
                f"Warning: alphas list length ({len(alphas)}) doesn't match number of datasets ({n_datasets}). Using default.")
            alphas = None

        if alphas is None:
            alphas = [0.6] * n_datasets

        # Process each x-variable separately (creates one figure per x-variable)
        for x_var in x_vars:
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Creating conditional histograms for X = {x_var}")
            self.logger.info(f"{'=' * 60}")

            # Collect all x-variable data to determine global bins
            all_x_data = []
            for df_name, df in dataframes.items():
                if x_var in df.columns:
                    all_x_data.extend(df[x_var].dropna().values)

            if not all_x_data:
                self.logger.warning(f"No data found for x-variable '{x_var}'. Skipping.")
                continue

            # Create bin edges for x-variable
            x_bin_edges = np.linspace(min(all_x_data), max(all_x_data), n_x_bins + 1)

            # Generate bin labels if not provided
            if x_bin_labels is None or len(x_bin_labels) != n_x_bins:
                x_bin_labels = [f'{x_var}: [{x_bin_edges[i]:.0f}, {x_bin_edges[i + 1]:.0f}]'
                                for i in range(n_x_bins)]

            # Create figure with subplots: rows = x_bins, cols = y_variables
            n_y_vars = len(y_variables)
            fig, axes = plt.subplots(n_x_bins, n_y_vars,
                                     figsize=(6 * n_y_vars, 4 * n_x_bins),
                                     squeeze=False)

            # Process each x-bin
            for bin_idx in range(n_x_bins):
                lower_bound = x_bin_edges[bin_idx]
                upper_bound = x_bin_edges[bin_idx + 1]

                # Include upper bound for last bin
                if bin_idx == n_x_bins - 1:
                    mask_func = lambda x: (x >= lower_bound) & (x <= upper_bound)
                else:
                    mask_func = lambda x: (x >= lower_bound) & (x < upper_bound)

                self.logger.info(f"\nProcessing bin {bin_idx + 1}/{n_x_bins}: {x_bin_labels[bin_idx]}")

                # Process each y-variable
                for y_idx, y_var in enumerate(y_variables):
                    ax = axes[bin_idx, y_idx]

                    # Collect data for this bin and y-variable from all datasets
                    bin_data_by_dataset = {}
                    all_y_values = []

                    for df_name, df in dataframes.items():
                        if x_var in df.columns and y_var in df.columns:
                            # Filter by x-bin
                            x_mask = mask_func(df[x_var])
                            y_values = df.loc[x_mask, y_var].dropna()

                            if len(y_values) > 0:
                                bin_data_by_dataset[df_name] = y_values
                                all_y_values.extend(y_values)

                                if print_stats:
                                    print(f"  {df_name} - {y_var}: {len(y_values)} points")

                    # Check if we have any data for this subplot
                    if not all_y_values:
                        ax.text(0.5, 0.5, 'No data available',
                                ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{y_var}\n{x_bin_labels[bin_idx]}\nNo Data', fontsize=9)
                        continue

                    # Determine histogram bin edges (common across datasets)
                    y_min, y_max = min(all_y_values), max(all_y_values)
                    hist_bin_edges = np.linspace(y_min, y_max, n_hist_bins + 1)

                    # Plot histograms for each dataset
                    if overlay_style == 'side_by_side' and len(bin_data_by_dataset) > 1:
                        # Calculate bar width and positions
                        bar_width = (hist_bin_edges[1] - hist_bin_edges[0]) / len(bin_data_by_dataset)

                        for dataset_idx, (df_name, y_data) in enumerate(bin_data_by_dataset.items()):
                            # Compute histogram
                            counts, _ = np.histogram(y_data, bins=hist_bin_edges, density=density)

                            # Calculate bar positions (offset for side-by-side)
                            bar_positions = hist_bin_edges[:-1] + dataset_idx * bar_width

                            ax.bar(bar_positions, counts, width=bar_width,
                                   alpha=alphas[dataset_names.index(df_name)],
                                   color=colors[dataset_names.index(df_name)],
                                   label=f'{df_name} (n={len(y_data)})',
                                   edgecolor='black', linewidth=0.5)

                            # Add statistics markers
                            if show_statistics:
                                mean_val = y_data.mean()
                                median_val = y_data.median()

                                y_height = counts.max() * 0.95
                                ax.plot(mean_val, y_height, 'v',
                                        color=colors[dataset_names.index(df_name)],
                                        markersize=8, label=f'{df_name} mean')
                                ax.plot(median_val, y_height * 0.90, '^',
                                        color=colors[dataset_names.index(df_name)],
                                        markersize=8, label=f'{df_name} median')

                    else:  # overlay style or single dataset
                        for df_name, y_data in bin_data_by_dataset.items():
                            ax.hist(y_data, bins=hist_bin_edges,
                                    alpha=alphas[dataset_names.index(df_name)],
                                    color=colors[dataset_names.index(df_name)],
                                    label=f'{df_name} (n={len(y_data)})',
                                    density=density,
                                    edgecolor='black', linewidth=0.5,
                                    stacked=stacked)

                            # Add KDE curve
                            if show_kde and len(y_data) > 1:
                                kde = stats.gaussian_kde(y_data)
                                x_range = np.linspace(y_min, y_max, 200)
                                kde_values = kde(x_range)

                                if not density:
                                    # Scale KDE to match histogram counts
                                    kde_values = kde_values * len(y_data) * (y_max - y_min) / n_hist_bins

                                ax.plot(x_range, kde_values,
                                        color=colors[dataset_names.index(df_name)],
                                        linewidth=2, alpha=0.8,
                                        linestyle='--')

                            # Add statistics
                            if show_statistics:
                                mean_val = y_data.mean()
                                median_val = y_data.median()

                                ax.axvline(mean_val,
                                           color=colors[dataset_names.index(df_name)],
                                           linestyle='--', linewidth=1.5, alpha=0.7)
                                ax.axvline(median_val,
                                           color=colors[dataset_names.index(df_name)],
                                           linestyle=':', linewidth=1.5, alpha=0.7)

                    # Labels and formatting
                    if bin_idx == n_x_bins - 1:  # Bottom row
                        ax.set_xlabel(y_var, fontsize=9)

                    if y_idx == 0:  # Leftmost column
                        ylabel = 'Density' if density else 'Frequency'
                        ax.set_ylabel(f'{ylabel}\n{x_bin_labels[bin_idx]}', fontsize=9)

                    # Title (only on top row)
                    if bin_idx == 0:
                        ax.set_title(f'{y_var}', fontsize=10, fontweight='bold')

                    ax.legend(fontsize=7, loc='best')
                    ax.grid(True, alpha=0.3, axis='y')

            # Overall figure title
            fig.suptitle(f'{name} - Conditional on {x_var}',
                         fontsize=14, fontweight='bold', y=0.995)
            plt.tight_layout()
            plt.show()

    def plotting_simple_multi_histograms(self,
                                         dataframes: dict[str, pd.DataFrame],
                                         y_variables: list[str],
                                         name: str = 'Multi-Dataset Histogram',
                                         n_hist_bins: int = 30,
                                         colors: list[str]|None = None,
                                         alphas: list[str]|None = None,
                                         use_different_colors: bool = True,
                                         density: bool = True,
                                         show_kde: bool = True,
                                         show_statistics: bool = True,
                                         print_stats: bool = False,
                                         overlay_style: str = 'overlay'):
        """
        Create simple histograms comparing distributions across multiple datasets.
        This is the unconditional version - just showing overall distributions without binning by x.

        CONCEPT: Simple multi-dataset histograms overlay or place side-by-side the distribution
        of the same variable across different datasets. This quickly shows:
        - Which dataset has higher/lower typical values
        - Which dataset has more spread/variability
        - Whether distributions have similar shapes

        Args:
            dataframes: Dictionary of {dataset_name: DataFrame}
            y_variables: Variables to plot (one subplot per variable)
            name: Base name for plot title
            n_hist_bins: Number of bins for each histogram
            colors: List of colors for each dataset
            alphas: List of transparency values for each dataset
            use_different_colors: Whether to use different colors per dataset
            density: If True, normalize histograms to probability density
            show_kde: Whether to overlay kernel density estimation curves
            show_statistics: Whether to display mean/median lines
            print_stats: Whether to print statistical summaries
            overlay_style: 'overlay' (bars overlapping) or 'side_by_side' (bars adjacent)
        """

        n_datasets = len(dataframes)
        dataset_names = list(dataframes.keys())

        # Setup colors
        if use_different_colors:
            if colors is None:
                import matplotlib.cm as cm
                colors = [cm.tab10(i % 10) for i in range(n_datasets)]
        else:
            colors = ['blue'] * n_datasets

        # Setup alphas
        if alphas is not None and len(alphas) != n_datasets:
            self.logger.warning(
                f"Warning: alphas list length ({len(alphas)}) doesn't match datasets. Using default.")
            alphas = None

        if alphas is None:
            alphas = [0.6] * n_datasets

        # Create subplot layout
        n_plots = len(y_variables)
        cols = math.ceil(math.sqrt(n_plots))
        rows = math.ceil(n_plots / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Process each y-variable
        for y_idx, y_var in enumerate(y_variables):
            ax = axes[y_idx]

            # Collect data from all datasets
            data_by_dataset = {}
            all_y_values = []

            for df_name, df in dataframes.items():
                if y_var in df.columns:
                    y_values = df[y_var].dropna()

                    if len(y_values) > 0:
                        data_by_dataset[df_name] = y_values
                        all_y_values.extend(y_values)

                        if print_stats:
                            print(f"\n{df_name} - {y_var}:")
                            print(f"  Count: {len(y_values)}")
                            print(f"  Mean: {y_values.mean():.2f}")
                            print(f"  Median: {y_values.median():.2f}")
                            print(f"  Std: {y_values.std():.2f}")

            if not all_y_values:
                ax.text(0.5, 0.5, 'No data available',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{y_var} - No Data')
                continue

            # Determine common bin edges
            y_min, y_max = min(all_y_values), max(all_y_values)
            hist_bin_edges = np.linspace(y_min, y_max, n_hist_bins + 1)

            # Plot histograms
            if overlay_style == 'side_by_side' and len(data_by_dataset) > 1:
                bar_width = (hist_bin_edges[1] - hist_bin_edges[0]) / len(data_by_dataset)

                for dataset_idx, (df_name, y_data) in enumerate(data_by_dataset.items()):
                    counts, _ = np.histogram(y_data, bins=hist_bin_edges, density=density)
                    bar_positions = hist_bin_edges[:-1] + dataset_idx * bar_width

                    ax.bar(bar_positions, counts, width=bar_width,
                           alpha=alphas[dataset_names.index(df_name)],
                           color=colors[dataset_names.index(df_name)],
                           label=f'{df_name} (n={len(y_data)})',
                           edgecolor='black', linewidth=0.5)

                    if show_statistics:
                        mean_val = y_data.mean()
                        median_val = y_data.median()
                        y_height = counts.max() * 0.95

                        ax.plot(mean_val, y_height, 'v',
                                color=colors[dataset_names.index(df_name)],
                                markersize=8)
                        ax.plot(median_val, y_height * 0.90, '^',
                                color=colors[dataset_names.index(df_name)],
                                markersize=8)

            else:  # overlay
                for df_name, y_data in data_by_dataset.items():
                    ax.hist(y_data, bins=hist_bin_edges,
                            alpha=alphas[dataset_names.index(df_name)],
                            color=colors[dataset_names.index(df_name)],
                            label=f'{df_name} (n={len(y_data)})',
                            density=density,
                            edgecolor='black', linewidth=0.5)

                    if show_kde and len(y_data) > 1:
                        kde = stats.gaussian_kde(y_data)
                        x_range = np.linspace(y_min, y_max, 200)
                        kde_values = kde(x_range)

                        if not density:
                            kde_values = kde_values * len(y_data) * (y_max - y_min) / n_hist_bins

                        ax.plot(x_range, kde_values,
                                color=colors[dataset_names.index(df_name)],
                                linewidth=2, alpha=0.8, linestyle='--')

                    if show_statistics:
                        mean_val = y_data.mean()
                        median_val = y_data.median()

                        ax.axvline(mean_val,
                                   color=colors[dataset_names.index(df_name)],
                                   linestyle='--', linewidth=1.5, alpha=0.7,
                                   label=f'{df_name} mean: {mean_val:.1f}')
                        ax.axvline(median_val,
                                   color=colors[dataset_names.index(df_name)],
                                   linestyle=':', linewidth=1.5, alpha=0.7,
                                   label=f'{df_name} median: {median_val:.1f}')

            ax.set_xlabel(y_var)
            ax.set_ylabel('Density' if density else 'Frequency')
            ax.set_title(FleetPlottingClass.wrap_title(f'{y_var} Distribution'))
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3, axis='y')

        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')

        fig.suptitle(f'{name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plotting_separate_histograms(self,
                                     dataframes: dict[str, pd.DataFrame],
                                     y_variables: list[str],
                                     name: str = 'Dataset Comparison',
                                     n_hist_bins: int = 30,
                                     variable_name_map: dict[str, str] | None = None,
                                     colors: list[str] = None,
                                     alphas: list[float] = None,
                                     use_different_colors: bool = True,
                                     density: bool = True,
                                     show_kde: bool = True,
                                     show_statistics: bool = True,
                                     print_stats: bool = False,
                                     share_x_axis: bool = True,
                                     share_y_axis: bool = False,
                                     single_figure: bool = True,
                                     transpose: bool = False,
                                     # ── layout ──────────────────────────────────────────────────
                                     cell_width: float = 6.0,
                                     cell_height: float = 4.0,
                                     gridspec_top: float = 0.93,
                                     # ── text ────────────────────────────────────────────────────
                                     base_font_size: int = 10,
                                     stats_font_size: int | None = None,
                                     suptitle_font_size: int | None = None,
                                     ):
        """
        Create separate histogram subplots for each dataset.

        Instead of overlaying all datasets on one plot, this creates a subplot
        for each dataset. This is preferable when you have many datasets (>3–4)
        that would clutter a single plot, or when you want to compare shapes,
        spreads, and statistics side-by-side without transparency overlap.

        Two figure-layout modes are available, selected via ``single_figure``:

        **``single_figure=False``** — one figure per y-variable.
            Example: 4 ship types × 3 variables → 3 separate figures, each
            with 4 subplots.

        **``single_figure=True``** — one figure containing all variables and
            all datasets.  Layout is rows × cols determined by ``transpose``:

            - ``transpose=False`` (default): rows = y-variables,
              cols = datasets.  Variables change down the page; ship types
              change across.
            - ``transpose=True``: rows = datasets, cols = y-variables.
              Ship types change down the page; variables change across.

        :param dataframes: Mapping of dataset label → DataFrame.  Each
            DataFrame must contain at least the columns listed in
            ``y_variables``.
        :type dataframes: dict[str, pd.DataFrame]
        :param y_variables: Column names to plot.  One subplot column (or row
            when transposed) is created per variable.
        :type y_variables: list[str]
        :param name: Base string used in the figure suptitle and per-variable
            figure titles.
        :type name: str
        :param n_hist_bins: Number of histogram bins.  When ``share_x_axis``
            is ``True`` the bins span the global min–max of each variable so
            that all subplots are directly comparable.
        :type n_hist_bins: int
        :param variable_name_map: Optional mapping from internal column names
            to publication-friendly display names, e.g.
            ``{'LOA (M)': 'LOA (m)', 'DWT': 'DWT (t)'}``.  Unmapped names
            are used as-is.
        :type variable_name_map: dict[str, str] or None
        :param colors: List of matplotlib colour specs, one per dataset.
            Overrides the automatic palette when provided.
        :type colors: list[str] or None
        :param alphas: List of bar transparency values (0–1), one per dataset.
            Defaults to 0.7 for all datasets when ``None``.
        :type alphas: list[float] or None
        :param use_different_colors: When ``True`` (default), each dataset
            receives a distinct colour from ``matplotlib.cm.tab10``.  Set to
            ``False`` to use blue for all datasets.
        :type use_different_colors: bool
        :param density: When ``True`` (default), histogram bars are normalised
            to a probability density so that bar areas sum to 1.  When
            ``False``, raw counts are shown.
        :type density: bool
        :param show_kde: Overlay a kernel density estimate curve on each
            histogram.  Requires at least two data points per dataset.
        :type show_kde: bool
        :param show_statistics: Draw vertical dashed/dotted lines for the
            mean and median, and show a statistics text box (sample size,
            mean, Anderson–Darling normality result).
        :type show_statistics: bool
        :param print_stats: Print per-dataset summary statistics (count, mean,
            median, std, range) and global ranges to stdout.  Useful for
            debugging without opening figures.
        :type print_stats: bool
        :param share_x_axis: When ``True`` (default), all subplots for the
            same variable share the same x-axis range (global min–max), making
            distributional differences immediately visible.
        :type share_x_axis: bool
        :param share_y_axis: When ``True``, all subplots in the same row share
            the same y-axis range, scaled to the row maximum.  Useful for
            comparing density heights across datasets.
        :type share_y_axis: bool
        :param single_figure: When ``True`` (default), all variables and
            datasets are combined into a single figure.  When ``False``, one
            figure is produced per y-variable.
        :type single_figure: bool
        :param transpose: Only applies when ``single_figure=True``.  When
            ``False`` (default) rows correspond to y-variables and columns to
            datasets.  When ``True`` the layout is transposed so rows
            correspond to datasets and columns to y-variables.
        :type transpose: bool
        :param cell_width: Width in inches of a single subplot cell.  Total
            figure width = ``cell_width × n_cols``.  Reducing this value is
            the primary lever for making the figure more compact
            horizontally.
        :type cell_width: float
        :param cell_height: Height in inches of a single subplot cell.  Total
            figure height = ``cell_height × n_rows``.  Reducing this value is
            the primary lever for making the figure more compact vertically.
        :type cell_height: float
        :param gridspec_top: Upper boundary of the subplot area as a fraction
            of the total figure height (0–1).  Controls the gap between the
            suptitle and the first row of panels.  Decrease when the suptitle
            is clipped; increase to reclaim vertical space for plots.  Applied
            via ``fig.subplots_adjust(top=gridspec_top)`` *after*
            ``tight_layout`` so it does not conflict with inter-panel spacing.
            Typical range 0.88–0.96.
        :type gridspec_top: float
        :param base_font_size: Base font size in points from which all other
            text elements are derived.  Axis labels use ``base_font_size``,
            column/row header titles use ``base_font_size + 1``, tick labels
            use ``base_font_size - 1``, and the secondary y-axis label
            (non-left-edge cells) uses ``base_font_size - 1``.  Changing
            this single value rescales all text consistently.
        :type base_font_size: int
        :param stats_font_size: Font size for the statistics text box shown
            in the top-right corner of each subplot.  When ``None``, defaults
            to ``base_font_size - 2``.  Set explicitly to keep the box
            readable on small cells without affecting other text.
        :type stats_font_size: int or None
        :param suptitle_font_size: Font size for the figure-level suptitle.
            When ``None``, defaults to ``base_font_size + 4``.  Set
            explicitly to increase title prominence independently of the
            rest of the text.
        :type suptitle_font_size: int or None
        """

        n_datasets = len(dataframes)
        dataset_names = list(dataframes.keys())

        # ── colour setup ────────────────────────────────────────────────────
        if use_different_colors:
            if colors is None:
                import matplotlib.cm as cm
                colors = [cm.tab10(i % 10) for i in range(n_datasets)]
        else:
            colors = ['blue'] * n_datasets

        # ── alpha setup ─────────────────────────────────────────────────────
        if alphas is not None and len(alphas) != n_datasets:
            self.logger.warning(
                f"Warning: alphas list length ({len(alphas)}) doesn't match "
                f"datasets. Using default.")
            alphas = None

        if alphas is None:
            alphas = [0.7] * n_datasets

        # ── derive font sizes from base, with optional overrides ────────────
        fs = {
            'axis_label':       base_font_size,
            'tick':             base_font_size - 1,
            'header':           base_font_size + 1,
            'ylabel_main':      base_font_size,
            'ylabel_secondary': base_font_size - 1,
            'stats':            stats_font_size    if stats_font_size    is not None
                                                   else base_font_size - 2,
            'suptitle':         suptitle_font_size if suptitle_font_size is not None
                                                   else base_font_size + 4,
        }

        if print_stats:
            print(f"[sizing] cell=({cell_width}×{cell_height}), "
                  f"gridspec_top={gridspec_top}, font sizes={fs}")

        # ====================================================================
        # SINGLE FIGURE MODE: all variables and datasets in one figure
        # ====================================================================
        if single_figure:
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info("Creating single figure with all variables and datasets")

            if transpose:
                self.logger.info(
                    f"Layout: {n_datasets} rows (datasets) × "
                    f"{len(y_variables)} cols (variables) [transposed]")
            else:
                self.logger.info(
                    f"Layout: {len(y_variables)} rows (variables) × "
                    f"{n_datasets} cols (datasets)")
            self.logger.info(f"{'=' * 60}")

            # ----------------------------------------------------------------
            # Stage 1: Collect data and compute global ranges per variable
            # ----------------------------------------------------------------
            if print_stats:
                print("\n[Stage 1] Collecting data and global ranges...")

            all_data = {}       # {y_var: {df_name: Series}}
            global_ranges = {}  # {y_var: (min, max)}

            for y_var in y_variables:
                all_data[y_var] = {}
                all_values = []

                for df_name, df in dataframes.items():
                    if y_var in df.columns:
                        y_values = df[y_var].dropna()
                        if len(y_values) > 0:
                            all_data[y_var][df_name] = y_values
                            all_values.extend(y_values)

                            if print_stats:
                                print(f"  {df_name} – {y_var}: n={len(y_values)}, "
                                      f"mean={y_values.mean():.2f}, "
                                      f"median={y_values.median():.2f}, "
                                      f"std={y_values.std():.2f}")

                if all_values:
                    global_ranges[y_var] = (min(all_values), max(all_values))
                    if print_stats:
                        print(f"  → global range for {y_var}: {global_ranges[y_var]}")

            # ----------------------------------------------------------------
            # Stage 2: Build figure grid based on transpose flag
            #
            # transpose=False (default):
            #   row_items = y_variables   → each row is one design variable
            #   col_items = dataset_names → each column is one ship type
            #
            # transpose=True:
            #   row_items = dataset_names → each row is one ship type
            #   col_items = y_variables   → each column is one design variable
            #
            # Inside the loop y_var and df_name are resolved from
            # row_item / col_item so all downstream plotting code is
            # identical in both orientations.
            # ----------------------------------------------------------------
            if print_stats:
                print("\n[Stage 2] Building figure grid...")

            if transpose:
                row_items = dataset_names
                col_items = y_variables
            else:
                row_items = y_variables
                col_items = dataset_names

            n_rows = len(row_items)
            n_cols = len(col_items)

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(cell_width * n_cols, cell_height * n_rows),
                squeeze=False
            )

            # Track per-row max height for optional y-axis sharing
            row_max_y = [0.0] * n_rows

            # ----------------------------------------------------------------
            # Stage 3: Populate each subplot
            # ----------------------------------------------------------------
            if print_stats:
                print("\n[Stage 3] Populating subplots...")

            for row_idx, row_item in enumerate(row_items):
                for col_idx, col_item in enumerate(col_items):

                    # Resolve which item maps to which role
                    if transpose:
                        df_name = row_item   # rows = ship types
                        y_var   = col_item   # cols = variables
                    else:
                        y_var   = row_item   # rows = variables
                        df_name = col_item   # cols = ship types

                    color_idx = dataset_names.index(df_name)
                    ax = axes[row_idx, col_idx]

                    y_name = (variable_name_map or {}).get(y_var, y_var)

                    if print_stats:
                        print(f"  Subplot [{row_idx},{col_idx}] → "
                              f"y_var={y_var}, dataset={df_name}")

                    # ---- No-data guards ------------------------------------
                    if y_var not in all_data or not all_data[y_var]:
                        ax.text(0.5, 0.5, 'No data available',
                                ha='center', va='center',
                                transform=ax.transAxes)
                        if col_idx == 0:
                            ax.set_ylabel(df_name if transpose else y_name,
                                          fontsize=fs['ylabel_main'],
                                          fontweight='bold')
                        if row_idx == 0:
                            ax.set_title(y_name if transpose else df_name,
                                         fontsize=fs['header'],
                                         fontweight='bold')
                        continue

                    if df_name not in all_data[y_var]:
                        ax.text(0.5, 0.5, 'No data',
                                ha='center', va='center',
                                transform=ax.transAxes)
                        if col_idx == 0:
                            ax.set_ylabel(df_name if transpose else y_name,
                                          fontsize=fs['ylabel_main'],
                                          fontweight='bold')
                        if row_idx == 0:
                            ax.set_title(y_name if transpose else df_name,
                                         fontsize=fs['header'],
                                         fontweight='bold',
                                         color=colors[color_idx])
                        continue

                    y_data = all_data[y_var][df_name]

                    # ---- Histogram bins ------------------------------------
                    if share_x_axis and y_var in global_ranges:
                        g_min, g_max = global_ranges[y_var]
                        hist_bins = np.linspace(g_min, g_max, n_hist_bins + 1)
                    else:
                        hist_bins = np.linspace(
                            y_data.min(), y_data.max(), n_hist_bins + 1)

                    # ---- Plot histogram ------------------------------------
                    counts, bin_edges, _ = ax.hist(
                        y_data,
                        bins=hist_bins,
                        alpha=alphas[color_idx],
                        color=colors[color_idx],
                        density=density,
                        edgecolor='black',
                        linewidth=0.5
                    )
                    row_max_y[row_idx] = max(row_max_y[row_idx], counts.max())

                    # ---- KDE curve ----------------------------------------
                    if show_kde and len(y_data) > 1:
                        kde = stats.gaussian_kde(y_data)
                        x_range = np.linspace(hist_bins[0], hist_bins[-1], 200)
                        kde_vals = kde(x_range)

                        if not density:
                            bin_w = hist_bins[1] - hist_bins[0]
                            kde_vals = kde_vals * len(y_data) * bin_w

                        ax.plot(x_range, kde_vals,
                                color=colors[color_idx],
                                linewidth=2.5, alpha=0.9, linestyle='-')

                    # ---- Statistics lines and text box --------------------
                    if show_statistics:
                        mean_val   = y_data.mean()
                        median_val = y_data.median()
                        ad_val     = self.anderson_normality(data=y_data)

                        ax.axvline(mean_val, color=colors[color_idx],
                                   linestyle='--', linewidth=1.5, alpha=0.7)
                        ax.axvline(median_val, color=colors[color_idx],
                                   linestyle=':', linewidth=1.5, alpha=0.7)

                        stats_text = (f'n={len(y_data)}'
                                      f'\nμ={mean_val:.1f}'
                                      f'\nA–D: {ad_val["statistic"]:.3f}'
                                      f'\nCrit: {ad_val["critical_value"]:.4f}'
                                      f'\nNormal? {ad_val["is_normal"]}')
                        ax.text(0.98, 0.97, stats_text,
                                transform=ax.transAxes,
                                va='top', ha='right',
                                bbox=dict(boxstyle='round',
                                          facecolor='wheat', alpha=0.6),
                                fontsize=fs['stats'])

                    # ---- Axis labels and column / row headers -------------
                    #
                    # x-axis label  = the variable name (regardless of transpose)
                    # left-edge ylabel = identifies what the ROW represents
                    # top-edge title   = identifies what the COLUMN represents
                    #
                    ax.set_xlabel(y_name, fontsize=fs['axis_label'])

                    if col_idx == 0:
                        ylabel_unit = 'Density' if density else 'Frequency'
                        row_label = df_name if transpose else y_name
                        ax.set_ylabel(f'{row_label}\n{ylabel_unit}',
                                      fontsize=fs['ylabel_main'],
                                      fontweight='bold')
                    else:
                        ax.set_ylabel('Density' if density else 'Frequency',
                                      fontsize=fs['ylabel_secondary'])

                    if row_idx == 0:
                        col_label = y_name if transpose else df_name
                        ax.set_title(col_label,
                                     fontsize=fs['header'],
                                     fontweight='bold',
                                     color='black')

                    ax.grid(True, alpha=0.3, axis='y')

            # ----------------------------------------------------------------
            # Stage 4: Optional shared y-axis, then render
            # ----------------------------------------------------------------
            if print_stats:
                print("\n[Stage 4] Applying shared y-axis and rendering...")

            if share_y_axis:
                for r in range(n_rows):
                    for c in range(n_cols):
                        axes[r, c].set_ylim(0, row_max_y[r] * 1.05)

            notes = []
            if share_x_axis:
                notes.append("shared x-axis per variable")
            note_str = f" ({', '.join(notes)})" if notes else ""

            fig.suptitle(f'{name}{note_str}',
                         fontsize=fs['suptitle'], fontweight='bold')
            plt.tight_layout()
            fig.subplots_adjust(top=gridspec_top)
            plt.show()
            return  # Exit — skip the per-variable figure loop below

        # ====================================================================
        # ORIGINAL MODE: separate figure per y-variable
        # ====================================================================
        for y_var in y_variables:
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Creating histogram comparison for {y_var}")
            self.logger.info(f"{'=' * 60}")

            y_name = (variable_name_map or {}).get(y_var, y_var)

            # Collect data from all datasets for this variable
            data_by_dataset = {}
            all_y_values    = []

            for df_name, df in dataframes.items():
                if y_var in df.columns:
                    y_values = df[y_var].dropna()

                    if len(y_values) > 0:
                        data_by_dataset[df_name] = y_values
                        all_y_values.extend(y_values)

                        if print_stats:
                            print(f"\n{df_name} - {y_name}:")
                            print(f"  Count:  {len(y_values)}")
                            print(f"  Mean:   {y_values.mean():.2f}")
                            print(f"  Median: {y_values.median():.2f}")
                            print(f"  Std:    {y_values.std():.2f}")
                            print(f"  Range:  [{y_values.min():.2f}, "
                                  f"{y_values.max():.2f}]")

            if not all_y_values:
                self.logger.warning(
                    f"No data found for variable '{y_var}'. Skipping.")
                continue

            # Global range for shared x-axis
            global_y_min  = min(all_y_values)
            global_y_max  = max(all_y_values)
            global_hist_bins = np.linspace(global_y_min, global_y_max,
                                           n_hist_bins + 1)

            # Subplot grid — max 2 columns for readability
            n_plots = len(data_by_dataset)
            cols    = min(2, n_plots)
            rows    = math.ceil(n_plots / cols)

            fig, axes = plt.subplots(
                rows, cols,
                figsize=(cell_width * cols, cell_height * rows),
                squeeze=False
            )
            axes = axes.flatten()

            global_y_axis_max = 0

            # First pass: plot all histograms and find max y value
            for dataset_idx, (df_name, y_data) in enumerate(
                    data_by_dataset.items()):
                ax = axes[dataset_idx]

                # Determine bins (shared or individual)
                if share_x_axis:
                    hist_bins = global_hist_bins
                else:
                    hist_bins = np.linspace(
                        y_data.min(), y_data.max(), n_hist_bins + 1)

                # Plot histogram
                counts, bin_edges, patches = ax.hist(
                    y_data,
                    bins=hist_bins,
                    alpha=alphas[dataset_names.index(df_name)],
                    color=colors[dataset_names.index(df_name)],
                    density=density,
                    edgecolor='black',
                    linewidth=0.5
                )
                global_y_axis_max = max(global_y_axis_max, counts.max())

                # KDE curve
                if show_kde and len(y_data) > 1:
                    kde = stats.gaussian_kde(y_data)

                    x_range = (np.linspace(global_y_min, global_y_max, 200)
                               if share_x_axis
                               else np.linspace(y_data.min(), y_data.max(), 200))

                    kde_values = kde(x_range)

                    if not density:
                        bin_width  = hist_bins[1] - hist_bins[0]
                        kde_values = kde_values * len(y_data) * bin_width

                    ax.plot(x_range, kde_values,
                            color=colors[dataset_names.index(df_name)],
                            linewidth=2.5, alpha=0.9,
                            linestyle='-', label='KDE')

                # Statistics lines
                if show_statistics:
                    mean_val   = y_data.mean()
                    median_val = y_data.median()

                    ax.axvline(mean_val,
                               color=colors[dataset_names.index(df_name)],
                               linestyle='--', linewidth=2, alpha=0.8,
                               label=f'Mean: {mean_val:.1f}')
                    ax.axvline(median_val,
                               color=colors[dataset_names.index(df_name)],
                               linestyle=':', linewidth=2, alpha=0.8,
                               label=f'Median: {median_val:.1f}')

                # Labels and formatting
                ax.set_xlabel(y_name, fontsize=fs['axis_label'])
                ax.set_ylabel('Density' if density else 'Frequency',
                              fontsize=fs['axis_label'])
                ax.set_title(f'{df_name} (n={len(y_data)})',
                             fontsize=fs['header'], fontweight='bold',
                             color=colors[dataset_names.index(df_name)])

                if show_statistics or show_kde:
                    ax.legend(fontsize=fs['tick'], loc='best', framealpha=0.9)

                ax.grid(True, alpha=0.3, axis='y')

                # Statistics text box
                stats_text = (f'μ={y_data.mean():.1f}'
                              f'\nσ={y_data.std():.1f}')
                ax.text(0.98, 0.97, stats_text,
                        transform=ax.transAxes,
                        verticalalignment='top',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round',
                                  facecolor='wheat', alpha=0.6),
                        fontsize=fs['stats'])

            # Second pass: apply shared y-axis if requested
            if share_y_axis:
                for dataset_idx in range(len(data_by_dataset)):
                    axes[dataset_idx].set_ylim(0, global_y_axis_max * 1.05)

            # Hide unused subplots
            for i in range(len(data_by_dataset), len(axes)):
                axes[i].axis('off')

            comparison_note = " (shared x-axis)" if share_x_axis else ""

            fig.suptitle(f'{name} - {y_name} Distribution{comparison_note}',
                         fontsize=fs['suptitle'], fontweight='bold')
            plt.tight_layout()
            fig.subplots_adjust(top=gridspec_top)
            plt.show()

    def plotting_separate_qqplots(self,
                                  dataframes: dict[str, pd.DataFrame],
                                  y_variables: list[str],
                                  name: str = 'Dataset Q-Q Plot Comparison',
                                  colors: list[str] = None,
                                  use_different_colors: bool = True,
                                  distribution: str = 'norm',
                                  reference_dataset: str = None,
                                  show_reference_line: bool = True,
                                  show_statistics: bool = True,
                                  print_stats: bool = False,
                                  single_figure: bool = True,
                                  standardize: bool = False):
        """
        Create separate Q-Q plot subplots for each dataset.

        CONCEPT: Q-Q (Quantile-Quantile) plots compare distributions by plotting quantiles.
        Two main modes:

        1. THEORETICAL Q-Q PLOT (distribution != None):
           - Compares each dataset against a theoretical distribution (default: normal)
           - Points on diagonal = data matches that distribution
           - Deviations reveal skewness, heavy tails, outliers
           - Use case: "Is my ship dimension data normally distributed?"

        2. EMPIRICAL Q-Q PLOT (reference_dataset != None):
           - Compares each dataset against a reference dataset
           - Points on diagonal = datasets have same distribution
           - Deviations show distribution differences
           - Use case: "Do alternative fuel ships follow same dimension distribution as conventional?"

        INTERPRETATION:
        - Points on diagonal line: distributions match
        - S-curve pattern: data is skewed relative to reference
        - Points above/below ends: heavy or light tails
        - Outliers: points far from the main trend

        UNDERSTANDING THE AXES:
        - By default, theoretical quantiles are standard (mean=0, std=1)
        - Your sample data stays in its original scale (e.g., 100-200m for ship length)
        - The diagonal line connects them using: sample = slope × theoretical + intercept
        - If you see a "vertical line" near zero, your data may have low variance
        - Set standardize=True to plot both axes in standard units for clearer patterns

        LAYOUT OPTIONS (same as histogram function):

        single_figure=True (default):
            - Creates ONE figure with all variables and datasets
            - Layout: rows=y_variables, cols=datasets
            - Each row shows all datasets for one variable

        single_figure=False:
            - Creates one figure per y-variable
            - Each figure has subplots for all datasets

        Args:
            dataframes: Dictionary of {dataset_name: DataFrame}
            y_variables: Variables to plot
            name: Base name for plot titles
            colors: List of colors for each dataset
            use_different_colors: Whether to use different colors per dataset
            distribution: Theoretical distribution ('norm', 'expon', 'uniform', 'lognorm', etc.)
                         Set to None to use reference_dataset mode instead
            reference_dataset: Name of dataset to use as reference (overrides distribution)
            show_reference_line: Whether to show the diagonal reference line
            show_statistics: Whether to display R² fit statistics
            print_stats: Whether to print statistical summaries
            single_figure: If True, create one figure with all variables (rows) and datasets (cols)
            standardize: If True, standardize data to mean=0, std=1 before plotting (clearer for low-variance data)
        """
        import scipy.stats as stats
        from scipy.stats import probplot
        import numpy as np
        import matplotlib.pyplot as plt
        import math

        n_datasets = len(dataframes)
        dataset_names = list(dataframes.keys())

        # Validate reference_dataset if provided
        if reference_dataset is not None:
            if reference_dataset not in dataset_names:
                self.logger.error(f"Reference dataset '{reference_dataset}' not found in dataframes")
                return
            mode = 'empirical'
        else:
            mode = 'theoretical'

        # Setup colors
        if use_different_colors:
            if colors is None:
                import matplotlib.cm as cm
                colors = [cm.tab10(i % 10) for i in range(n_datasets)]
        else:
            colors = ['blue'] * n_datasets

        # SINGLE FIGURE MODE: All variables and datasets in one figure
        if single_figure:
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Creating single figure with Q-Q plots")
            self.logger.info(f"Mode: {mode.upper()}")
            if mode == 'theoretical':
                self.logger.info(f"Distribution: {distribution}")
            else:
                self.logger.info(f"Reference dataset: {reference_dataset}")
            self.logger.info(f"Layout: {len(y_variables)} rows × {n_datasets} columns")
            self.logger.info(f"{'=' * 60}")

            # Collect data for all variables
            all_data = {}  # {y_var: {df_name: data}}
            reference_data = {}  # {y_var: reference quantiles}

            for y_var in y_variables:
                all_data[y_var] = {}

                for df_name, df in dataframes.items():
                    if y_var in df.columns:
                        y_values = df[y_var].dropna()
                        if len(y_values) > 0:
                            all_data[y_var][df_name] = y_values

                            if print_stats:
                                print(f"\n{df_name} - {y_var}:")
                                print(f"  Count: {len(y_values)}")
                                print(f"  Mean: {y_values.mean():.2f}")
                                print(f"  Median: {y_values.median():.2f}")
                                print(f"  Std: {y_values.std():.2f}")
                                print(f"  Skewness: {stats.skew(y_values):.3f}")
                                print(f"  Kurtosis: {stats.kurtosis(y_values):.3f}")

                # For empirical mode, get reference quantiles
                if mode == 'empirical' and reference_dataset in all_data[y_var]:
                    reference_data[y_var] = np.sort(all_data[y_var][reference_dataset])

            # Create single figure with grid layout
            n_rows = len(y_variables)
            n_cols = n_datasets
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(6 * n_cols, 5 * n_rows),
                                     squeeze=False)

            # Plot each variable (row) × dataset (column) combination
            for row_idx, y_var in enumerate(y_variables):
                if y_var not in all_data or not all_data[y_var]:
                    # No data for this variable
                    for col_idx in range(n_cols):
                        ax = axes[row_idx, col_idx]
                        ax.text(0.5, 0.5, 'No data available',
                                ha='center', va='center', transform=ax.transAxes)
                        if col_idx == 0:
                            ax.set_ylabel(y_var, fontsize=11, fontweight='bold')
                        if row_idx == 0:
                            ax.set_title(dataset_names[col_idx], fontsize=11, fontweight='bold')
                    continue

                # Plot each dataset in this row
                for col_idx, df_name in enumerate(dataset_names):
                    ax = axes[row_idx, col_idx]

                    if df_name not in all_data[y_var]:
                        ax.text(0.5, 0.5, 'No data',
                                ha='center', va='center', transform=ax.transAxes)
                        if col_idx == 0:
                            ax.set_ylabel(y_var, fontsize=11, fontweight='bold')
                        if row_idx == 0:
                            ax.set_title(df_name, fontsize=11, fontweight='bold',
                                         color=colors[col_idx])
                        continue

                    y_data = all_data[y_var][df_name]

                    if mode == 'theoretical':
                        # CONCEPT: probplot compares data to theoretical distribution
                        # It returns theoretical quantiles (standard scale) and ordered sample values
                        # The fit line (slope, intercept, r) shows how well data matches after scaling

                        # Optionally standardize data first for clearer visualization
                        if standardize:
                            y_data_plot = (y_data - y_data.mean()) / y_data.std()
                        else:
                            y_data_plot = y_data

                        # Theoretical Q-Q plot
                        (theoretical_quantiles, ordered_values), (slope, intercept, r) = \
                            probplot(y_data_plot, dist=distribution, plot=None)

                        # Plot points - these are already on the correct scale
                        ax.scatter(theoretical_quantiles, ordered_values,
                                   alpha=0.6, color=colors[col_idx], s=30,
                                   edgecolors='black', linewidth=0.5)

                        # Reference line - this is the ideal fit line
                        if show_reference_line:
                            ax.plot(theoretical_quantiles,
                                    slope * theoretical_quantiles + intercept,
                                    'r--', linewidth=2, alpha=0.7, label='Ideal fit')

                        # Labels
                        xlabel = f'Theoretical Quantiles ({distribution})'
                        if standardize:
                            ylabel = 'Standardized Sample Quantiles'
                        else:
                            ylabel = 'Sample Quantiles'

                        # Calculate R²
                        r_squared = r ** 2

                    else:
                        # Empirical Q-Q plot against reference dataset
                        if y_var not in reference_data:
                            ax.text(0.5, 0.5, 'No reference data',
                                    ha='center', va='center', transform=ax.transAxes)
                            continue

                        ref_quantiles = reference_data[y_var]
                        sample_quantiles = np.sort(y_data)

                        # Interpolate to match lengths
                        n_ref = len(ref_quantiles)
                        n_sample = len(sample_quantiles)

                        if n_sample < n_ref:
                            # Interpolate sample to reference length
                            x_sample = np.linspace(0, 1, n_sample)
                            x_ref = np.linspace(0, 1, n_ref)
                            sample_interp = np.interp(x_ref, x_sample, sample_quantiles)
                            x_values = ref_quantiles
                            y_values = sample_interp
                        else:
                            # Interpolate reference to sample length
                            x_ref = np.linspace(0, 1, n_ref)
                            x_sample = np.linspace(0, 1, n_sample)
                            ref_interp = np.interp(x_sample, x_ref, ref_quantiles)
                            x_values = ref_interp
                            y_values = sample_quantiles

                        # Plot points
                        ax.scatter(x_values, y_values,
                                   alpha=0.6, color=colors[col_idx], s=30,
                                   edgecolors='black', linewidth=0.5)

                        # Reference line (y=x)
                        if show_reference_line:
                            min_val = min(x_values.min(), y_values.min())
                            max_val = max(x_values.max(), y_values.max())
                            ax.plot([min_val, max_val], [min_val, max_val],
                                    'r--', linewidth=2, alpha=0.7, label='y=x')

                        # Calculate correlation
                        r_squared = np.corrcoef(x_values, y_values)[0, 1] ** 2

                        # Labels
                        xlabel = f'{reference_dataset} Quantiles'
                        ylabel = f'{df_name} Quantiles'

                    # Add statistics
                    if show_statistics:
                        stats_text = f'n={len(y_data)}\nR²={r_squared:.3f}'
                        ax.text(0.02, 0.98, stats_text,
                                transform=ax.transAxes,
                                verticalalignment='top',
                                horizontalalignment='left',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                                fontsize=9)

                    # Formatting
                    ax.set_xlabel(xlabel, fontsize=10)

                    # Only show y-label on leftmost column
                    if col_idx == 0:
                        ax.set_ylabel(f'{y_var}\n{ylabel}', fontsize=10, fontweight='bold')
                    else:
                        ax.set_ylabel(ylabel, fontsize=10)

                    # Only show title on top row
                    if row_idx == 0:
                        ax.set_title(df_name, fontsize=11, fontweight='bold',
                                     color=colors[col_idx])

                    ax.grid(True, alpha=0.3)
                    if show_reference_line:
                        ax.legend(fontsize=8, loc='best')

            # Overall figure title
            mode_description = f"vs {distribution} distribution" if mode == 'theoretical' else f"vs {reference_dataset}"
            fig.suptitle(f'{name} ({mode_description})',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()

            return

        # ORIGINAL MODE: Separate figure per y-variable
        for y_var in y_variables:
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Creating Q-Q plots for {y_var}")
            self.logger.info(f"Mode: {mode.upper()}")
            self.logger.info(f"{'=' * 60}")

            # Collect data
            data_by_dataset = {}
            for df_name, df in dataframes.items():
                if y_var in df.columns:
                    y_values = df[y_var].dropna()
                    if len(y_values) > 0:
                        data_by_dataset[df_name] = y_values

            if not data_by_dataset:
                self.logger.warning(f"No data found for variable '{y_var}'. Skipping.")
                continue

            # Get reference data for empirical mode
            ref_quantiles = None
            if mode == 'empirical':
                if reference_dataset in data_by_dataset:
                    ref_quantiles = np.sort(data_by_dataset[reference_dataset])
                else:
                    self.logger.warning(f"Reference dataset '{reference_dataset}' has no data for '{y_var}'. Skipping.")
                    continue

            # Calculate layout
            n_plots = len(data_by_dataset)
            cols = min(2, n_plots)
            rows = math.ceil(n_plots / cols)

            # Create figure
            fig, axes = plt.subplots(rows, cols,
                                     figsize=(7 * cols, 6 * rows),
                                     squeeze=False)
            axes = axes.flatten()

            # Plot each dataset
            for dataset_idx, (df_name, y_data) in enumerate(data_by_dataset.items()):
                ax = axes[dataset_idx]

                if mode == 'theoretical':
                    # CONCEPT: probplot compares data to theoretical distribution
                    # By default, keeps data in original scale with fit line showing the transformation
                    # With standardize=True, converts both to standard units for clearer patterns

                    # Optionally standardize data first
                    if standardize:
                        y_data_plot = (y_data - y_data.mean()) / y_data.std()
                    else:
                        y_data_plot = y_data

                    # Theoretical Q-Q plot
                    (theoretical_quantiles, ordered_values), (slope, intercept, r) = \
                        probplot(y_data_plot, dist=distribution, plot=None)

                    ax.scatter(theoretical_quantiles, ordered_values,
                               alpha=0.6, color=colors[dataset_names.index(df_name)],
                               s=40, edgecolors='black', linewidth=0.5)

                    if show_reference_line:
                        ax.plot(theoretical_quantiles,
                                slope * theoretical_quantiles + intercept,
                                'r--', linewidth=2.5, alpha=0.8, label='Ideal fit')

                    xlabel = f'Theoretical Quantiles ({distribution})'
                    if standardize:
                        ylabel = 'Standardized Sample Quantiles'
                    else:
                        ylabel = 'Sample Quantiles'
                    r_squared = r ** 2

                else:
                    # Empirical Q-Q plot
                    sample_quantiles = np.sort(y_data)
                    n_ref = len(ref_quantiles)
                    n_sample = len(sample_quantiles)

                    if n_sample < n_ref:
                        x_sample = np.linspace(0, 1, n_sample)
                        x_ref = np.linspace(0, 1, n_ref)
                        sample_interp = np.interp(x_ref, x_sample, sample_quantiles)
                        x_values = ref_quantiles
                        y_values = sample_interp
                    else:
                        x_ref = np.linspace(0, 1, n_ref)
                        x_sample = np.linspace(0, 1, n_sample)
                        ref_interp = np.interp(x_sample, x_ref, ref_quantiles)
                        x_values = ref_interp
                        y_values = sample_quantiles

                    ax.scatter(x_values, y_values,
                               alpha=0.6, color=colors[dataset_names.index(df_name)],
                               s=40, edgecolors='black', linewidth=0.5)

                    if show_reference_line:
                        min_val = min(x_values.min(), y_values.min())
                        max_val = max(x_values.max(), y_values.max())
                        ax.plot([min_val, max_val], [min_val, max_val],
                                'r--', linewidth=2.5, alpha=0.8, label='y=x')

                    r_squared = np.corrcoef(x_values, y_values)[0, 1] ** 2
                    xlabel = f'{reference_dataset} Quantiles'
                    ylabel = f'{df_name} Quantiles'

                # Formatting
                ax.set_xlabel(xlabel, fontsize=11)
                ax.set_ylabel(ylabel, fontsize=11)
                ax.set_title(f'{df_name} (n={len(y_data)})',
                             fontsize=12, fontweight='bold',
                             color=colors[dataset_names.index(df_name)])

                if show_statistics:
                    stats_text = f'R²={r_squared:.3f}\nμ={y_data.mean():.1f}\nσ={y_data.std():.1f}'
                    ax.text(0.02, 0.98, stats_text,
                            transform=ax.transAxes,
                            verticalalignment='top',
                            horizontalalignment='left',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                            fontsize=9)

                ax.grid(True, alpha=0.3)
                if show_reference_line:
                    ax.legend(fontsize=9, loc='best')

            # Hide unused subplots
            for i in range(len(data_by_dataset), len(axes)):
                axes[i].axis('off')

            # Figure title
            mode_description = f"vs {distribution} distribution" if mode == 'theoretical' else f"vs {reference_dataset}"
            fig.suptitle(f'{name} - {y_var} Q-Q Plot ({mode_description})',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()

    def plotting_bootstrap_ci_comparison(self,
                                         dataframes: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
                                         y_variables: list[str],
                                         name: str = 'Bootstrap Confidence Interval Comparison',
                                         n_bootstrap: int = 10000,
                                         confidence_level: float = 0.95,
                                         statistic: str = 'both',  # 'mean', 'median', or 'both'
                                         colors: list[str] = None,
                                         use_different_colors: bool = True,
                                         show_zero_line: bool = True,
                                         show_observed: bool = True,
                                         n_hist_bins: int = 50,
                                         print_stats: bool = False,
                                         single_figure: bool = True):
        """
        Create bootstrap confidence interval plots comparing conventional vs alternative fuel ships.

        CONCEPT: Bootstrap resampling quantifies uncertainty in differences between datasets.

        How it works:
        1. For each ship type and dimension, we have two datasets: conventional and alternative
        2. We repeatedly resample each dataset (with replacement) many times (e.g., 10,000)
        3. For each resample, calculate the difference in mean/median
        4. This creates a distribution of differences (the bootstrap distribution)
        5. The 95% confidence interval is the 2.5th to 97.5th percentile of this distribution

        INTERPRETATION:
        - If the CI excludes zero: Significant difference between fuel types
        - If the CI includes zero: No significant difference
        - CI width: Uncertainty in the estimate (narrower = more certain)
        - Observed difference (red line): The actual difference in your data

        PRACTICAL USE:
        - "Are LNG tankers significantly longer than conventional tankers?"
        - "How much larger is the beam of methanol ships, and how certain are we?"
        - Bootstrap works even when data isn't normally distributed

        LAYOUT OPTIONS:

        single_figure=True (default):
            - One figure per statistic type (median/mean)
            - Layout: rows=y_variables, cols=ship_types
            - Compare all ship types for each dimension side-by-side

        single_figure=False:
            - One figure per y-variable
            - Each figure shows all ship types as subplots
            - Useful when you have many variables or need separate exports

        Args:
            dataframes: Dictionary mapping ship_type -> (conventional_df, alternative_df)
                       Example: {'Tanker': (tanker_conv_df, tanker_alt_df),
                                'Bulk': (bulk_conv_df, bulk_alt_df)}
            y_variables: List of dimension variables to analyze
            name: Base name for plot titles
            n_bootstrap: Number of bootstrap resamples (10000 recommended)
            confidence_level: Confidence level for intervals (default 0.95 = 95%)
            statistic: Which statistic to plot: 'mean', 'median', or 'both'
            colors: List of colors for each ship type
            use_different_colors: Whether to use different colors per ship type
            show_zero_line: Whether to show vertical line at zero (no difference)
            show_observed: Whether to show the observed difference as red dashed line
            n_hist_bins: Number of bins for bootstrap distribution histogram
            print_stats: Whether to print detailed statistics
            single_figure: If True, create one figure per statistic with all variables and ship types
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import math

        n_ship_types = len(dataframes)
        ship_type_names = list(dataframes.keys())

        # Validate statistic parameter
        if statistic not in ['mean', 'median', 'both']:
            self.logger.error(f"statistic must be 'mean', 'median', or 'both', got '{statistic}'")
            return

        # Setup colors
        if use_different_colors:
            if colors is None:
                import matplotlib.cm as cm
                colors = [cm.tab10(i % 10) for i in range(n_ship_types)]
        else:
            colors = ['steelblue'] * n_ship_types

        # Determine which statistics to compute
        compute_median = statistic in ['median', 'both']
        compute_mean = statistic in ['mean', 'both']

        # SINGLE FIGURE MODE
        if single_figure:
            # Create separate figures for median and mean if both requested
            statistics_to_plot = []
            if compute_median:
                statistics_to_plot.append('median')
            if compute_mean:
                statistics_to_plot.append('mean')

            for stat_type in statistics_to_plot:
                self.logger.info(f"\n{'=' * 60}")
                self.logger.info(f"Creating single figure for {stat_type} bootstrap CIs")
                self.logger.info(f"Layout: {len(y_variables)} rows × {n_ship_types} columns")
                self.logger.info(f"Bootstrap samples: {n_bootstrap}, CI: {confidence_level * 100}%")
                self.logger.info(f"{'=' * 60}")

                # Collect all bootstrap results
                all_results = {}  # {y_var: {ship_type: bootstrap_results}}

                for y_var in y_variables:
                    all_results[y_var] = {}

                    for ship_type, (conv_df, alt_df) in dataframes.items():
                        # Get clean data
                        if y_var not in conv_df.columns or y_var not in alt_df.columns:
                            continue

                        conv_data = conv_df[y_var].dropna()
                        alt_data = alt_df[y_var].dropna()

                        if len(conv_data) < 3 or len(alt_data) < 3:
                            self.logger.warning(f"Insufficient data for {ship_type} - {y_var}")
                            continue

                        # Perform bootstrap
                        bootstrap_diffs = []

                        for _ in range(n_bootstrap):
                            conv_resample = np.random.choice(conv_data, size=len(conv_data), replace=True)
                            alt_resample = np.random.choice(alt_data, size=len(alt_data), replace=True)

                            if stat_type == 'median':
                                diff = np.median(alt_resample) - np.median(conv_resample)
                            else:  # mean
                                diff = np.mean(alt_resample) - np.mean(conv_resample)

                            bootstrap_diffs.append(diff)

                        bootstrap_diffs = np.array(bootstrap_diffs)

                        # Calculate CI
                        alpha = 1 - confidence_level
                        lower_percentile = (alpha / 2) * 100
                        upper_percentile = (1 - alpha / 2) * 100

                        ci = (
                            np.percentile(bootstrap_diffs, lower_percentile),
                            np.percentile(bootstrap_diffs, upper_percentile)
                        )

                        # Observed difference
                        if stat_type == 'median':
                            observed_diff = np.median(alt_data) - np.median(conv_data)
                        else:
                            observed_diff = np.mean(alt_data) - np.mean(conv_data)

                        # Store results
                        all_results[y_var][ship_type] = {
                            'bootstrap_diffs': bootstrap_diffs,
                            'ci': ci,
                            'observed_diff': observed_diff,
                            'n_conv': len(conv_data),
                            'n_alt': len(alt_data),
                            'significant': ci[0] > 0 or ci[1] < 0
                        }

                        if print_stats:
                            print(f"\n{ship_type} - {y_var} ({stat_type}):")
                            print(f"  Observed difference: {observed_diff:.2f}")
                            print(f"  95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
                            print(f"  Significant: {'Yes' if ci[0] > 0 or ci[1] < 0 else 'No'}")
                            print(f"  n_conv: {len(conv_data)}, n_alt: {len(alt_data)}")

                # Create figure
                n_rows = len(y_variables)
                n_cols = n_ship_types
                fig, axes = plt.subplots(n_rows, n_cols,
                                         figsize=(6 * n_cols, 5 * n_rows),
                                         squeeze=False)

                # Plot each variable (row) × ship type (column)
                for row_idx, y_var in enumerate(y_variables):
                    if y_var not in all_results or not all_results[y_var]:
                        # No data for this variable
                        for col_idx in range(n_cols):
                            ax = axes[row_idx, col_idx]
                            ax.text(0.5, 0.5, 'No data available',
                                    ha='center', va='center', transform=ax.transAxes)
                            if col_idx == 0:
                                ax.set_ylabel(y_var, fontsize=11, fontweight='bold')
                            if row_idx == 0:
                                ax.set_title(ship_type_names[col_idx], fontsize=11, fontweight='bold')
                        continue

                    for col_idx, ship_type in enumerate(ship_type_names):
                        ax = axes[row_idx, col_idx]

                        if ship_type not in all_results[y_var]:
                            ax.text(0.5, 0.5, 'No data',
                                    ha='center', va='center', transform=ax.transAxes)
                            if col_idx == 0:
                                ax.set_ylabel(y_var, fontsize=11, fontweight='bold')
                            if row_idx == 0:
                                ax.set_title(ship_type, fontsize=11, fontweight='bold',
                                             color=colors[col_idx])
                            continue

                        results = all_results[y_var][ship_type]

                        # Plot histogram of bootstrap distribution
                        ax.hist(results['bootstrap_diffs'], bins=n_hist_bins,
                                alpha=0.7, color=colors[col_idx],
                                edgecolor='black', linewidth=0.5)

                        # Add zero line (no difference)
                        if show_zero_line:
                            ax.axvline(0, color='black', linestyle='-',
                                       linewidth=1.5, alpha=0.5, label='No difference')

                        # Add observed difference
                        if show_observed:
                            ax.axvline(results['observed_diff'], color='red',
                                       linestyle='--', linewidth=2,
                                       label=f"Observed: {results['observed_diff']:.1f}")

                        # Add confidence interval bounds
                        ax.axvline(results['ci'][0], color='green',
                                   linestyle=':', linewidth=2, alpha=0.8)
                        ax.axvline(results['ci'][1], color='green',
                                   linestyle=':', linewidth=2, alpha=0.8)

                        # Shade CI region
                        ylim = ax.get_ylim()
                        ax.axvspan(results['ci'][0], results['ci'][1],
                                   alpha=0.2, color='green',
                                   label=f"{int(confidence_level * 100)}% CI")

                        # Labels and formatting
                        if row_idx == n_rows - 1:
                            ax.set_xlabel(
                                f'Difference in {stat_type.capitalize()} {y_var}\n(Alternative - Conventional)',
                                fontsize=9)

                        if col_idx == 0:
                            ax.set_ylabel(f'{y_var}\nFrequency', fontsize=10, fontweight='bold')
                        else:
                            ax.set_ylabel('Frequency', fontsize=9)

                        if row_idx == 0:
                            ax.set_title(ship_type,
                                         fontsize=11, fontweight='bold',
                                         color='black')

                        # Add statistics text box
                        sig_text = 'SIG' if results['significant'] else 'n.s.'
                        stats_text = (f"n_alt={results['n_alt']}, n_conv={results['n_conv']}\n"
                                      f"CI: [{results['ci'][0]:.1f}, {results['ci'][1]:.1f}]\n"
                                      f"{sig_text}")

                        ax.text(0.98, 0.97, stats_text,
                                transform=ax.transAxes,
                                verticalalignment='top',
                                horizontalalignment='right',
                                bbox=dict(boxstyle='round',
                                          facecolor='lightgreen' if results['significant'] else 'wheat',
                                          alpha=0.7),
                                fontsize=8)

                        ax.grid(True, alpha=0.3, axis='y')

                        if row_idx == 0 and col_idx == 0:
                            ax.legend(fontsize=8, loc='upper left')

                # Overall title
                fig.suptitle(f'{name} - {stat_type.capitalize()} Differences\n'
                             f'(Bootstrap n={n_bootstrap}, CI={int(confidence_level * 100)}%)',
                             fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show()

            return

        # ORIGINAL MODE: Separate figure per y-variable
        for y_var in y_variables:
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Creating bootstrap CI plots for {y_var}")
            self.logger.info(f"{'=' * 60}")

            # Collect results for this variable
            variable_results = {}

            for ship_type, (conv_df, alt_df) in dataframes.items():
                if y_var not in conv_df.columns or y_var not in alt_df.columns:
                    continue

                conv_data = conv_df[y_var].dropna()
                alt_data = alt_df[y_var].dropna()

                if len(conv_data) < 3 or len(alt_data) < 3:
                    self.logger.warning(f"Insufficient data for {ship_type} - {y_var}")
                    continue

                # Perform bootstrap for both statistics if requested
                results = {}

                if compute_median:
                    bootstrap_median_diffs = []
                    for _ in range(n_bootstrap):
                        conv_resample = np.random.choice(conv_data, size=len(conv_data), replace=True)
                        alt_resample = np.random.choice(alt_data, size=len(alt_data), replace=True)
                        diff = np.median(alt_resample) - np.median(conv_resample)
                        bootstrap_median_diffs.append(diff)

                    bootstrap_median_diffs = np.array(bootstrap_median_diffs)
                    alpha = 1 - confidence_level
                    median_ci = (
                        np.percentile(bootstrap_median_diffs, (alpha / 2) * 100),
                        np.percentile(bootstrap_median_diffs, (1 - alpha / 2) * 100)
                    )
                    observed_median = np.median(alt_data) - np.median(conv_data)

                    results['median'] = {
                        'bootstrap_diffs': bootstrap_median_diffs,
                        'ci': median_ci,
                        'observed': observed_median,
                        'significant': median_ci[0] > 0 or median_ci[1] < 0
                    }

                if compute_mean:
                    bootstrap_mean_diffs = []
                    for _ in range(n_bootstrap):
                        conv_resample = np.random.choice(conv_data, size=len(conv_data), replace=True)
                        alt_resample = np.random.choice(alt_data, size=len(alt_data), replace=True)
                        diff = np.mean(alt_resample) - np.mean(conv_resample)
                        bootstrap_mean_diffs.append(diff)

                    bootstrap_mean_diffs = np.array(bootstrap_mean_diffs)
                    alpha = 1 - confidence_level
                    mean_ci = (
                        np.percentile(bootstrap_mean_diffs, (alpha / 2) * 100),
                        np.percentile(bootstrap_mean_diffs, (1 - alpha / 2) * 100)
                    )
                    observed_mean = np.mean(alt_data) - np.mean(conv_data)

                    results['mean'] = {
                        'bootstrap_diffs': bootstrap_mean_diffs,
                        'ci': mean_ci,
                        'observed': observed_mean,
                        'significant': mean_ci[0] > 0 or mean_ci[1] < 0
                    }

                results['n_conv'] = len(conv_data)
                results['n_alt'] = len(alt_data)
                variable_results[ship_type] = results

                if print_stats:
                    print(f"\n{ship_type} - {y_var}:")
                    if 'median' in results:
                        print(f"  Median difference: {results['median']['observed']:.2f}")
                        print(f"    95% CI: [{results['median']['ci'][0]:.2f}, {results['median']['ci'][1]:.2f}]")
                        print(f"    Significant: {results['median']['significant']}")
                    if 'mean' in results:
                        print(f"  Mean difference: {results['mean']['observed']:.2f}")
                        print(f"    95% CI: [{results['mean']['ci'][0]:.2f}, {results['mean']['ci'][1]:.2f}]")
                        print(f"    Significant: {results['mean']['significant']}")

            if not variable_results:
                self.logger.warning(f"No data for {y_var}, skipping")
                continue

            # Determine layout
            n_plots = len(variable_results)
            n_stats = int(compute_median) + int(compute_mean)

            # Create figure(s) based on statistic parameter
            if statistic == 'both':
                # Create one figure with median on left, mean on right for each ship type
                cols = 2
                rows = n_plots
                fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows), squeeze=False)

                for row_idx, (ship_type, results) in enumerate(variable_results.items()):
                    for col_idx, stat_type in enumerate(['median', 'mean']):
                        ax = axes[row_idx, col_idx]
                        stat_results = results[stat_type]

                        # Plot histogram
                        ax.hist(stat_results['bootstrap_diffs'], bins=n_hist_bins,
                                alpha=0.7, color=colors[ship_type_names.index(ship_type)],
                                edgecolor='black', linewidth=0.5)

                        if show_zero_line:
                            ax.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)

                        if show_observed:
                            ax.axvline(stat_results['observed'], color='red',
                                       linestyle='--', linewidth=2,
                                       label=f"Observed: {stat_results['observed']:.1f}")

                        ax.axvline(stat_results['ci'][0], color='green', linestyle=':', linewidth=2)
                        ax.axvline(stat_results['ci'][1], color='green', linestyle=':', linewidth=2)

                        ylim = ax.get_ylim()
                        ax.axvspan(stat_results['ci'][0], stat_results['ci'][1],
                                   alpha=0.2, color='green')

                        ax.set_xlabel(f'Difference in {stat_type.capitalize()} (Alt - Conv)', fontsize=10)
                        ax.set_ylabel('Frequency', fontsize=10)

                        ax.set_title(f'{ship_type} - {stat_type.capitalize()}',
                                     fontsize=11, fontweight='bold',
                                     color='black')

                        stats_text = (f"n={results['n_alt']}/{results['n_conv']}\n"
                                      f"CI: [{stat_results['ci'][0]:.1f}, {stat_results['ci'][1]:.1f}]")
                        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                                va='top', ha='right',
                                bbox=dict(boxstyle='round',
                                          facecolor='lightgreen' if stat_results['significant'] else 'wheat',
                                          alpha=0.7),
                                fontsize=8)

                        ax.grid(True, alpha=0.3, axis='y')
                        ax.legend(fontsize=8)

                fig.suptitle(f'{name} - {y_var}\nBootstrap n={n_bootstrap}, CI={int(confidence_level * 100)}%',
                             fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show()

            else:
                # Single statistic - one column layout
                stat_type = statistic
                rows = n_plots
                cols = 1
                fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows), squeeze=False)
                axes = axes.flatten()

                for idx, (ship_type, results) in enumerate(variable_results.items()):
                    ax = axes[idx]
                    stat_results = results[stat_type]

                    ax.hist(stat_results['bootstrap_diffs'], bins=n_hist_bins,
                            alpha=0.7, color=colors[ship_type_names.index(ship_type)],
                            edgecolor='black', linewidth=0.5)

                    if show_zero_line:
                        ax.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5,
                                   label='No difference')

                    if show_observed:
                        ax.axvline(stat_results['observed'], color='red',
                                   linestyle='--', linewidth=2,
                                   label=f"Observed: {stat_results['observed']:.1f}")

                    ax.axvline(stat_results['ci'][0], color='green', linestyle=':', linewidth=2)
                    ax.axvline(stat_results['ci'][1], color='green', linestyle=':', linewidth=2)

                    ax.axvspan(stat_results['ci'][0], stat_results['ci'][1],
                               alpha=0.2, color='green',
                               label=f"{int(confidence_level * 100)}% CI")

                    ax.set_xlabel(f'Difference in {stat_type.capitalize()} (Alt - Conv)', fontsize=11)
                    ax.set_ylabel('Frequency', fontsize=11)

                    ax.set_title(f'{ship_type} (n_alt={results["n_alt"]}, n_conv={results["n_conv"]})',
                                 fontsize=12, fontweight='bold',
                                 color='black')

                    stats_text = f"CI: [{stat_results['ci'][0]:.1f}, {stat_results['ci'][1]:.1f}]"
                    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                            va='top', ha='right',
                            bbox=dict(boxstyle='round',
                                      facecolor='lightgreen' if stat_results['significant'] else 'wheat',
                                      alpha=0.7),
                            fontsize=9)

                    ax.grid(True, alpha=0.3, axis='y')
                    ax.legend(fontsize=9, loc='best')

                fig.suptitle(f'{name} - {y_var} ({stat_type.capitalize()} Difference)\n'
                             f'Bootstrap n={n_bootstrap}, CI={int(confidence_level * 100)}%',
                             fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show()