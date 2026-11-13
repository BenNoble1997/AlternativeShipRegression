import logging
import pandas as pd
import numpy as np
from typing import Union, List, Dict
import folium

from pathlib import Path
import plotly.express as px
from matplotlib import pyplot as plt
import plotly.graph_objects as go  # Only needed for the port circles
from datetime import datetime

from config.logger_config import LoggerMixin
from Operational_profile.data_processing.plotting_functions.Data_visualiser_class import GenericDataVisualizer

class ShipRoutePlotter(LoggerMixin):

    def __init__(self,
                 verbose: bool = True,
                 debug: bool = False):
        self.verbose = verbose
        self.debug = debug

        if self.verbose:
            self.logger.setLevel(logging.INFO)
        elif self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.ERROR)


    def _plot_route_single(self,
                           title: str,
                           datafile: pd.DataFrame,
                           output_folder: str|Path,
                           filetitle: str = None,
                           name: str = None,
                           timestamp_column: str = None,
                           lat_column: str = None,
                           lon_column: str = None,
                           colour_column: str = None,
                           colour_column_name: str = None,
                           start_time: str|datetime|None = None,
                           end_time: str|datetime|None = None,
                           map_style: str = "carto_voyager",
                   ):
        """
                Create a single route plot for a dataframe.

                :param title: Title to display on the plot.
                :type title: str
                :param filetitle: Part of filename for the saved HTML file.
                :type filetitle: str
                :param datafile: DataFrame containing the data to plot.
                :type datafile: pandas.DataFrame
                :param name: Identifier for the plot and filename.
                :type name: str
                :param output_folder: Folder to save the HTML file.
                :type output_folder: str
                :param timestamp_column: Column name for timestamp data.
                :type timestamp_column: str
                :param lat_column: Column name for latitude data.
                :type lat_column: str
                :param lon_column: Column name for longitude data.
                :type lon_column: str
                :param colour_column: Column name for data used to color points.
                :type colour_column: str
                :param colour_column_name: Label for the color scale.
                :type colour_column_name: str
                :param map_style: Map style to use. posibilities: "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner"
                """

        if isinstance(output_folder, str):
            output_folder_path = Path(output_folder)
        else:
            output_folder_path = output_folder

        if name is None:
            name = 'Unnamed ship'

        if filetitle is None:
            filetitle = name

        # Filter dataframe by time range if specified
        filtered_datafile = datafile.copy()

        if timestamp_column is not None and (start_time is not None or end_time is not None):
            # Ensure timestamp column is datetime
            if not pd.api.types.is_datetime64_any_dtype(filtered_datafile[timestamp_column]):
                filtered_datafile[timestamp_column] = pd.to_datetime(filtered_datafile[timestamp_column])

            # Apply time filtering
            if start_time is not None:
                start_time_dt = pd.to_datetime(start_time)
                filtered_datafile = filtered_datafile[filtered_datafile[timestamp_column] >= start_time_dt]

            if end_time is not None:
                end_time_dt = pd.to_datetime(end_time)
                filtered_datafile = filtered_datafile[filtered_datafile[timestamp_column] <= end_time_dt]

        # Check if filtered dataframe is empty
        if filtered_datafile.empty:
            print(f"Warning: No data found in the specified time range for {name}")
            return

        file_name = output_folder_path / f'{name}_sailing_route_{filetitle}.html'

        hover_name = timestamp_column if timestamp_column is not None else None

        fig = px.scatter_map(filtered_datafile,
                                lat=lat_column,
                                lon=lon_column,
                                hover_name=hover_name,
                                color=colour_column,
                                zoom=3,
                                labels={colour_column: colour_column_name},
                                # height=800,
                                # width=800,
                                )

        # fig.update_layout(map_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0})

        fig.update_layout(
            map=dict(
                style=map_style,
                # Calculate bounds with some padding
                bounds=dict(
                    west=filtered_datafile[lon_column].min() - 1,
                    east=filtered_datafile[lon_column].max() + 1,
                    south=filtered_datafile[lat_column].min() - 1,
                    north=filtered_datafile[lat_column].max() + 1
                )
            )
        )

        # Use the correct column access syntax
        if timestamp_column is not None and not filtered_datafile.empty:
            actual_start_time = filtered_datafile[timestamp_column].iloc[0]
            actual_end_time = filtered_datafile[timestamp_column].iloc[-1]
        else:
            actual_start_time = "N/A"
            actual_end_time = "N/A"

        fig.update_layout(
            title_text=f"{name}: between {actual_start_time} and {actual_end_time}; {title}")
        # plot(fig)

        fig.write_html(file_name)

        # Debug: Print file path to verify
        self.logger.info(f"Attempting to save file to: {file_name}")

        fig.write_html(file_name)

        # Verify file was created
        if file_name.exists():
            self.logger.info(f"File successfully created: {file_name}")
        else:
            self.logger.info(f"File was NOT created: {file_name}")

    def plot_route(self,
                   title: str,
                   datafile: pd.DataFrame,
                   filetitle: str = None,
                   name: str=None,
                   output_folder: str|Path = None,
                    timestamp_column:str = None,
                    lat_column:str = None,
                    lon_column:str = None,
                    colour_column:str = None,
                    colour_column_name:str = None,
                   start_time: str|datetime|None = None,
                   end_time: str|datetime|None = None,
                   map_style: str = "carto-positron",
                   ):

        if isinstance(datafile,dict):
            for key, df in datafile.items():
                title = f"{key}_{title}"
                filetitle = f"{key}_{filetitle}"
                self._plot_route_single(title=title,
                                        filetitle=filetitle,
                                        datafile=df,
                                        name=name,
                                        output_folder=output_folder,
                                        timestamp_column=timestamp_column,
                                        lat_column=lat_column,
                                        lon_column=lon_column,
                                        colour_column=colour_column,
                                        colour_column_name=colour_column_name,
                                        start_time=start_time,
                                        end_time=end_time,
                                        map_style=map_style,)

        else:
            if name is None:
                name = 'Ship1'
            else:
                name = name
            self._plot_route_single(title=title,
                                    filetitle=filetitle,
                                    datafile = datafile,
                                    name=name,
                                    output_folder=output_folder,
                                    timestamp_column=timestamp_column,
                                    lat_column=lat_column,
                                    lon_column=lon_column,
                                    colour_column=colour_column,
                                    colour_column_name=colour_column_name,
                                    start_time=start_time,
                                    end_time=end_time,
                                    map_style=map_style,)

    def _plot_route_with_ports_single(self,
                                      title: str,
                                      output_folder: str|Path,
                                      datafile: pd.DataFrame = None,
                                      ports_dataframe: pd.DataFrame = None,
                                      filetitle: str = None,
                                      name: str = None,
                                      timestamp_column: str = None,
                                      lat_column: str = None,
                                      lon_column: str = None,
                                      colour_column: str = None,
                                      colour_column_name: str = None,
                                      port_id_column: str = 'PORT_ID',
                                      port_lat_column: str = 'LATITUDE',
                                      port_lon_column: str = 'LONGITUDE',
                                      port_duration_column: str = 'TOTAL_DURATION_HOURS',
                                      port_type_column: str = None,  # NEW: Column indicating if port is from database
                                      start_time: str|datetime|None = None,
                                      end_time: str|datetime|None = None,
                                      ):
        """
        Create a plot with optional route and/or port overlays.

        NEW: Supports coloring ports by type (database vs discovered)

        :param port_type_column: Column name indicating if port is from database (True/False)
        :type port_type_column: str
        """

        # Validate input - at least one dataframe must be provided
        if datafile is None and ports_dataframe is None:
            raise ValueError("At least one of 'datafile' or 'ports_dataframe' must be provided")

        if isinstance(output_folder, str):
            output_folder_path = Path(output_folder)
        else:
            output_folder_path = output_folder

        if name is None:
            name = 'Map_Plot'

        if filetitle is None:
            filetitle = name

        # Initialize variables for bounds calculation
        all_lats = []
        all_lons = []

        # Handle route data filtering and processing (unchanged)
        filtered_datafile = None
        if datafile is not None and not datafile.empty:
            filtered_datafile = datafile.copy()

            if timestamp_column is not None and (start_time is not None or end_time is not None):
                # Ensure timestamp column is datetime
                if not pd.api.types.is_datetime64_any_dtype(filtered_datafile[timestamp_column]):
                    filtered_datafile[timestamp_column] = pd.to_datetime(filtered_datafile[timestamp_column])

                # Apply time filtering
                if start_time is not None:
                    start_time_dt = pd.to_datetime(start_time)
                    filtered_datafile = filtered_datafile[filtered_datafile[timestamp_column] >= start_time_dt]

                if end_time is not None:
                    end_time_dt = pd.to_datetime(end_time)
                    filtered_datafile = filtered_datafile[filtered_datafile[timestamp_column] <= end_time_dt]

            # Check if filtered dataframe is empty
            if filtered_datafile.empty:
                print(f"Warning: No route data found in the specified time range for {name}")
                filtered_datafile = None
            else:
                all_lats.extend(filtered_datafile[lat_column].tolist())
                all_lons.extend(filtered_datafile[lon_column].tolist())

        # Add port coordinates to bounds calculation
        if ports_dataframe is not None and not ports_dataframe.empty:
            all_lats.extend(ports_dataframe[port_lat_column].tolist())
            all_lons.extend(ports_dataframe[port_lon_column].tolist())

        # Check if we have any data to plot
        if not all_lats or not all_lons:
            print(f"Warning: No data to plot for {name}")
            return

        file_name = output_folder_path / f'{name}_map_plot_{filetitle}.html'

        # Create the main figure - start with route data if available
        if filtered_datafile is not None:
            hover_name = timestamp_column if timestamp_column is not None else None

            # Create figure using px.scatter_mapbox like your original code
            fig = px.scatter_map(filtered_datafile,
                                    lat=lat_column,
                                    lon=lon_column,
                                    hover_name=hover_name,
                                    color=colour_column,
                                    zoom=3,
                                    labels={colour_column: colour_column_name} if colour_column else {},
                                    )
        else:
            # If no route data, create empty figure with first port as center
            fig = px.scatter_map(
                pd.DataFrame({
                    'lat': [ports_dataframe[port_lat_column].iloc[0]],
                    'lon': [ports_dataframe[port_lon_column].iloc[0]]
                }),
                lat='lat',
                lon='lon',
                zoom=3
            )
            # Clear the trace since we only used it for positioning
            fig.data = []

        # === NEW: Enhanced port plotting with type-based coloring ===
        if ports_dataframe is not None and not ports_dataframe.empty:
            # Prepare port data
            ports_df = ports_dataframe.copy()

            # Calculate circle sizes based on duration if provided
            if port_duration_column is not None and port_duration_column in ports_df.columns:
                # Normalize duration to reasonable circle sizes (adjust scale as needed)
                min_duration = ports_df[port_duration_column].min()
                max_duration = ports_df[port_duration_column].max()

                if max_duration > min_duration:
                    # Scale to range 15-60 for visibility
                    ports_df['circle_size'] = 15 + 45 * (ports_df[port_duration_column] - min_duration) / (
                            max_duration - min_duration)
                else:
                    ports_df['circle_size'] = 30  # Default size if all durations are the same

                # Create hover text with duration info
                ports_df['hover_text'] = ports_df[port_id_column].astype(str) + '<br>Duration: ' + ports_df[
                    port_duration_column].astype(str)
            else:
                # Default circle size if no duration column
                ports_df['circle_size'] = 30
                ports_df['hover_text'] = ports_df[port_id_column].astype(str)

            # === NEW: Add port type information to hover text ===
            if port_type_column is not None and port_type_column in ports_df.columns:
                port_type_text = ports_df[port_type_column].apply(
                    lambda x: 'Database Port' if x else 'Discovered Port'
                )
                ports_df['hover_text'] = ports_df['hover_text'] + '<br>Type: ' + port_type_text

            # === NEW: Plot ports by type with different colors ===
            if port_type_column is not None and port_type_column in ports_df.columns:
                # Split ports by type
                db_ports = ports_df[ports_df[port_type_column] == True]
                discovered_ports = ports_df[ports_df[port_type_column] == False]

                # Add database ports (green)
                if not db_ports.empty:
                    fig.add_trace(
                        go.Scattermap(
                            lat=db_ports[port_lat_column],
                            lon=db_ports[port_lon_column],
                            mode='markers',
                            marker=dict(
                                size=db_ports['circle_size'],
                                color='rgba(0, 150, 0, 0.8)',  # Green for database ports
                                sizemode='diameter'
                                # Note: Scattermapbox doesn't support 'line' property for borders
                            ),
                            text=db_ports['hover_text'],
                            hovertemplate='<b>Port:</b> %{text}<extra></extra>',
                            name='Database Ports',
                            showlegend=True
                        )
                    )

                # Add discovered ports (red)
                if not discovered_ports.empty:
                    fig.add_trace(
                        go.Scattermap(
                            lat=discovered_ports[port_lat_column],
                            lon=discovered_ports[port_lon_column],
                            mode='markers',
                            marker=dict(
                                size=discovered_ports['circle_size'],
                                color='rgba(255, 0, 0, 0.8)',  # Red for discovered ports
                                sizemode='diameter'
                                # Note: Scattermapbox doesn't support 'line' property for borders
                            ),
                            text=discovered_ports['hover_text'],
                            hovertemplate='<b>Port:</b> %{text}<extra></extra>',
                            name='Discovered Ports',
                            showlegend=True
                        )
                    )
            else:
                # === FALLBACK: Original single-color plotting ===
                fig.add_trace(
                    go.Scattermap(
                        lat=ports_df[port_lat_column],
                        lon=ports_df[port_lon_column],
                        mode='markers',
                        marker=dict(
                            size=ports_df['circle_size'],
                            color='rgba(255, 0, 0, 0.6)',  # Default red
                            sizemode='diameter'
                        ),
                        text=ports_df['hover_text'],
                        hovertemplate='<b>Port:</b> %{text}<extra></extra>',
                        name='Ports',
                        showlegend=True
                    )
                )

        # Update layout (unchanged)
        fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0})

        # Calculate bounds
        fig.update_layout(
            map=dict(
                style="open-street-map",
                bounds=dict(
                    west=min(all_lons) - 1,
                    east=max(all_lons) + 1,
                    south=min(all_lats) - 1,
                    north=max(all_lats) + 1
                )
            )
        )

        # Set title
        if filtered_datafile is not None and timestamp_column is not None:
            actual_start_time = filtered_datafile[timestamp_column].iloc[0]
            actual_end_time = filtered_datafile[timestamp_column].iloc[-1]
            title_suffix = f": between {actual_start_time} and {actual_end_time}"
        else:
            title_suffix = ""

        fig.update_layout(title_text=f"{name}{title_suffix}; {title}")

        # Save the file
        fig.write_html(file_name)

        # Debug: Print file path to verify
        self.logger.info(f"Attempting to save file to: {file_name}")

        # Verify file was created
        if file_name.exists():
            self.logger.info(f"File successfully created: {file_name}")
        else:
            self.logger.info(f"File was NOT created: {file_name}")

    def plot_route_with_ports(self,
                              title: str,
                              output_folder: str|Path = None,
                              datafile: pd.DataFrame = None,
                              ports_dataframe: pd.DataFrame = None,
                              filetitle: str = None,
                              name: str = None,
                              timestamp_column: str = None,
                              lat_column: str = None,
                              lon_column: str = None,
                              colour_column: str = None,
                              colour_column_name: str = None,
                              port_id_column: str = 'PORT_ID',
                              port_lat_column: str = 'LATITUDE',
                              port_lon_column: str = 'LONGITUDE',
                              port_duration_column: str = 'TOTAL_DURATION_HOURS',
                              port_type_column: str = None,  # NEW: Column indicating if port is from database
                              start_time: str|datetime|None = None,
                              end_time: str|datetime|None = None,
                              ):
        """
        Create map plots with flexible combinations of route and/or port data.

        NEW: Supports coloring ports by type (database vs discovered)

        Usage examples:
        - Route only: provide datafile, set ports_dataframe=None
        - Ports only: provide ports_dataframe, set datafile=None
        - Route + Ports: provide both datafile and ports_dataframe
        - Colored by port type: provide port_type_column='EXISTING_PORT'

        :param port_type_column: Column name indicating if port is from database (True/False)
        :type port_type_column: str
        """

        if isinstance(datafile, dict):
            for key, df in datafile.items():
                title_modified = f"{key}_{title}"
                filetitle_modified = f"{key}_{filetitle}" if filetitle else key
                self._plot_route_with_ports_single(
                    title=title_modified,
                    filetitle=filetitle_modified,
                    datafile=df,
                    ports_dataframe=ports_dataframe,
                    name=name,
                    output_folder=output_folder,
                    timestamp_column=timestamp_column,
                    lat_column=lat_column,
                    lon_column=lon_column,
                    colour_column=colour_column,
                    colour_column_name=colour_column_name,
                    port_id_column=port_id_column,
                    port_lat_column=port_lat_column,
                    port_lon_column=port_lon_column,
                    port_duration_column=port_duration_column,
                    port_type_column=port_type_column,  # NEW: Pass through port type column
                    start_time=start_time,
                    end_time=end_time,
                )
        else:
            if name is None:
                if datafile is not None and ports_dataframe is not None:
                    name = 'Route_and_Ports'
                elif datafile is not None:
                    name = 'Route_Only'
                elif ports_dataframe is not None:
                    name = 'Ports_Only'
                else:
                    name = 'Map_Plot'

            self._plot_route_with_ports_single(
                title=title,
                filetitle=filetitle,
                datafile=datafile,
                ports_dataframe=ports_dataframe,
                name=name,
                output_folder=output_folder,
                timestamp_column=timestamp_column,
                lat_column=lat_column,
                lon_column=lon_column,
                colour_column=colour_column,
                colour_column_name=colour_column_name,
                port_id_column=port_id_column,
                port_lat_column=port_lat_column,
                port_lon_column=port_lon_column,
                port_duration_column=port_duration_column,
                port_type_column=port_type_column,  # NEW: Pass through port type column
                start_time=start_time,
                end_time=end_time,
            )

    # Updated convenience methods
    def plot_route_only(self, **kwargs):
        """Plot route data only - convenience method"""
        kwargs['ports_dataframe'] = None
        return self.plot_route_with_ports(**kwargs)

    def plot_ports_only(self, **kwargs):
        """Plot ports data only - convenience method"""
        kwargs['datafile'] = None
        return self.plot_route_with_ports(**kwargs)

    # Updated get_ports_summary function to use new coloring


    def _plot_folium_map_single(self,
                        datafile: pd.DataFrame,
                        name: str = None,
                        output_folder: Union[str,Path] = None,
                        ):
        """
        Plots a folium map of the time series  data

        requires a LATITUDE and LONGITUDE columns to be viable

        Credit: Roy de Winter
        :param datafile: DataFrame containing the data to plot.
        :param name: name of the plot
        :return:
        """
        def plotShips(point):
            """for each coordinate plot a Circle on a folium map."""
            folium.Circle(location=[point.LATITUDE, point.LONGITUDE],
                          radius=20,
                          weight=1).add_to(world_map)
        world_map = folium.Map(prefer_canvas = (True))

        datafile.apply(plotShips, axis=1)

        world_map.fit_bounds(world_map.get_bounds())

        if name is None:
            name = 'Unnamed ship'

        if isinstance(output_folder, str):
            output_folder_path = Path(output_folder)
        else:
            output_folder_path = output_folder or Path.cwd()

        output_folder_path.mkdir(parents=True, exist_ok=True)

        file_name = output_folder_path / f'{name}_sailing_route.html'

        print(f"Attempting to save to: {file_name}")
        print(f"Directory exists: {output_folder_path.exists()}")
        print(f"Full absolute path: {file_name.absolute()}")

        world_map.save(str(file_name))

        # Verify file was created
        if file_name.exists():
            self.logger.info(f"File successfully saved: {file_name}")
        else:
            self.logger.error(f"File was NOT created: {file_name}")

    def plot_folium_map(self,
                        datafile: Union[Dict[str,pd.DataFrame], pd.DataFrame],
                        name: str = None,
                        combine: bool = False,
                        output_folder: Union[str,Path] = None):

        if isinstance(datafile, dict):
            for key, df in datafile.items():
                name = f"{key}_{name}" if name else key
                self._plot_folium_map_single(datafile=df,
                                             name=name,
                                             output_folder=output_folder)

        else:
            self._plot_folium_map_single(datafile=datafile,
                                         name=name,
                                         output_folder=output_folder)

    def plot_time_segment(self,
                          datafiles: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                               timestamp_column: str,
                               y_columns: Union[List[str], str],
                               figsize=(15, 12),
                               cols=2,
                               start_time: datetime = None,
                               end_time: datetime = None,
                          fill_area: Union[bool, Dict[str, bool]] = False,
                          fill_alpha: float = 0.3,
                               **kwargs):
        """
        Create subplots for multiple y-axis columns, handling both single DataFrames and dictionaries of DataFrames.

        When provided with a dictionary of DataFrames, it filters to only plot data from DataFrames
        that contain the specified start_time and/or end_time range.

        :param datafiles: Single DataFrame or dictionary of DataFrames
        :param timestamp_column: Name of the timestamp column
        :param y_columns: List of column names to plot separately
        :param figsize: Overall figure size
        :param cols: Number of columns in subplot grid
        :param start_time: Start time filter (optional)
        :param end_time: End time filter (optional)
        :param fill_area: Whether to fill area under curves. Can be:
                         - bool: applies to all lines
                         - dict: maps column names to bool values for selective filling
        :param fill_alpha: Alpha transparency for filled areas (0-1)
        :param kwargs: Additional arguments to pass to time_series_plot
        :return: Figure and axes objects
        :raises ValueError: If no datafiles contain the specified time range
        """

        if isinstance(y_columns,str):
            y_columns = [y_columns]

        # Handle single DataFrame case
        if isinstance(datafiles, pd.DataFrame):
            return self.plot_time_segment_single(
                datafile=datafiles,
                timestamp_column=timestamp_column,
                y_columns=y_columns,
                figsize=figsize,
                cols=cols,
                start_time=start_time,
                end_time=end_time,
                fill_area=fill_area,
                fill_alpha=fill_alpha,
                **kwargs
            )

        # Handle dictionary of DataFrames
        if not isinstance(datafiles, dict):
            raise TypeError("datafiles must be either a pandas DataFrame or a dictionary of DataFrames")



        # If no time filters specified, plot all datafiles
        if start_time is None and end_time is None:
            matching_datafiles = datafiles
        else:
            # Find datafiles that contain the specified time range
            matching_datafiles = {}

            for name, df in datafiles.items():
                if timestamp_column not in df.columns:
                    continue

                # Ensure timestamp column is datetime
                if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
                    try:
                        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
                    except:
                        continue

                # Get min and max timestamps in this datafile
                df_min_time = df[timestamp_column].min()
                df_max_time = df[timestamp_column].max()

                # Check if this datafile overlaps with the requested time range
                overlaps = True

                if start_time is not None and df_max_time < start_time:
                    overlaps = False

                if end_time is not None and df_min_time > end_time:
                    overlaps = False

                if overlaps:
                    matching_datafiles[name] = df

        # Check if any datafiles match the criteria
        if not matching_datafiles:
            time_range_str = ""
            if start_time and end_time:
                time_range_str = f" between {start_time} and {end_time}"
            elif start_time:
                time_range_str = f" after {start_time}"
            elif end_time:
                time_range_str = f" before {end_time}"

            available_ranges = []
            for name, df in datafiles.items():
                if timestamp_column in df.columns:
                    try:
                        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
                            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
                        min_time = df[timestamp_column].min()
                        max_time = df[timestamp_column].max()
                        available_ranges.append(f"{name}: {min_time} to {max_time}")
                    except:
                        available_ranges.append(f"{name}: invalid timestamp data")

            raise ValueError(f"No datafiles contain data{time_range_str}.\n"
                             f"Available time ranges:\n" + "\n".join(available_ranges))

        # Calculate total subplots needed (y_columns × number of matching datafiles)
        total_plots = len(y_columns) * len(matching_datafiles)

        # Calculate grid dimensions
        rows = (total_plots + cols - 1) // cols

        # Create the main figure and subplots
        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        # Handle single subplot case
        if rows == 1 and cols ==1:
            axes_flat = [axes]
        elif rows == 1:
            # Single row - ensure it's iterable
            if isinstance(axes, np.ndarray):
                axes_flat = axes.flatten()  # Convert 1D array to flat array
            else:
                axes_flat = [axes]  # Single axes object

        else:
            axes_flat = axes.flatten()

        plot_idx = 0

        # Plot each datafile and y-column combination
        for datafile_name, df in matching_datafiles.items():
            for y_col in y_columns:
                if plot_idx < len(axes_flat):
                    # Create visualizer for this datafile
                    visualiser = GenericDataVisualizer(data=df)

                    # Create title with datafile name
                    title = f'{y_col} Over Time - {datafile_name}'

                    # Use the visualizer's method with specific axis
                    visualiser.time_series_plot(
                        time_column=timestamp_column,
                        y_columns=y_col,
                        ax=axes_flat[plot_idx],
                        title=title,
                        y_label=y_col,
                        start_time=start_time,
                        end_time=end_time,
                        fill_area=fill_area,
                        fill_alpha = fill_alpha,
                        **kwargs
                    )

                    plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes_flat)):
            axes_flat[i].set_visible(False)

        plt.tight_layout()
        return fig, axes

    def plot_time_segment_single(self,
                                 datafile: pd.DataFrame,
                                 timestamp_column: str,
                                 y_columns: Union[List[str], str],
                                 figsize=(15, 12),
                                 cols=2,
                                 start_time: datetime = None,
                                 end_time: datetime = None,
                                 fill_area: Union[bool, Dict[str, bool]] = False,
                                 fill_alpha: float = 0.3,
                                 **kwargs):
        """
        Helper function for plotting a single DataFrame (original functionality).
        """
        visualiser = GenericDataVisualizer(data=datafile)

        if isinstance(y_columns,str):
            y_columns = [y_columns]

        # Calculate rows needed
        rows = (len(y_columns) + cols - 1) // cols

        # Create the main figure and subplots
        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        # Handle single subplot case
        if rows == 1 and cols ==1:
            # Single subplot - axes is already a single Axes object
            axes_flat = [axes]
        elif rows == 1:
            # Single row - axes is a 1D array
            if isinstance(axes,np.ndarray):
                axes_flat = axes.flatten()
            else:
                axes_flat = [axes]

        else:
            # Multiple rows - flatten the 2D array
            axes_flat = axes.flatten()  #

        # Plot each y-column in its own subplot
        for i, y_col in enumerate(y_columns):
            if i < len(axes_flat):
                # Use the visualizer's method with specific axis

                #DEBUG
                current_ax = axes_flat[i]
                print(f"DEBUG: axes_flat[{i}] type: {type(current_ax)}")
                print(f"DEBUG: axes_flat[{i}] value: {current_ax}")

                visualiser.time_series_plot(
                    time_column=timestamp_column,
                    y_columns=y_col,
                    ax=axes_flat[i],
                    title=f'{y_col} Over Time',
                    y_label=y_col,
                    start_time=start_time,
                    end_time=end_time,
                    fill_area=fill_area,
                    fill_alpha=fill_alpha,
                    **kwargs
                )

        # Hide unused subplots
        for i in range(len(y_columns), len(axes_flat)):
            axes_flat[i].set_visible(False)

        plt.tight_layout()
        return fig, axes