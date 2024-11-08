import os
import glob
import pandas as pd
import re
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import dash_bootstrap_components as dbc
import paramiko
import stat
from dotenv import load_dotenv        
import requests
import time
import json

# Cache file paths
CACHE_DIR = 'cache'
CACHE_FILE_CURRENTS = os.path.join(CACHE_DIR, 'currents_data.json')
CACHE_FILE_WAVES = os.path.join(CACHE_DIR, 'waves_data.json')
CACHE_FILE_WIND = os.path.join(CACHE_DIR, 'wind_data.json')
CACHE_EXPIRY_TIME = 3600  # Cache expiry time in seconds (1 hour)

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

# Function to check if cache is expired
def is_cache_expired(cache_file):
    if not os.path.exists(cache_file):
        return True
    cache_time = os.path.getmtime(cache_file)
    current_time = time.time()
    return (current_time - cache_time) > CACHE_EXPIRY_TIME

# Function to load data from cache if available
def load_cache_data(cache_file):
    with open(cache_file, 'r') as f:
        return pd.DataFrame(json.load(f))

# Function to save data to cache
def save_to_cache(data, cache_file):
    with open(cache_file, 'w') as f:
        json.dump(data.to_dict(orient='records'), f)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Fetch data functions with cache checking
def fetch_data(endpoint, cache_file):
    if is_cache_expired(cache_file):
        url = f"{API_BASE_URL}/data/{endpoint}"
        response = requests.get(url)
        data = pd.DataFrame(response.json())
        save_to_cache(data, cache_file)
    else:
        data = load_cache_data(cache_file)
    return data

# Load initial data (with cache mechanism)
currents_data = fetch_data("currents", CACHE_FILE_CURRENTS)
waves_data = fetch_data("waves", CACHE_FILE_WAVES)
wind_data = fetch_data("wind", CACHE_FILE_WIND)

# Load currents data
# currents_data = process_currents_data(CURRENT_DATA_DIR) # comment if you are loading fron the folder
currents_data['Datetime'] = pd.to_datetime(currents_data['Datetime'])

# Load waves data
# waves_data = process_wave_data(WAVE_DATA_DIR) # comment if you are loading fron the folder
waves_data['Datetime'] = pd.to_datetime(waves_data['Datetime'])

# List of columns for currents dropdown
currents_columns = ["Velocity 1 (m/s)", "Velocity 2 (m/s)", "Velocity 3 (m/s)", "Speed (m/s)", "Direction (°)",
                    "Amplitude units", "Correlation 1 (%)", "Correlation 2 (%)", "Correlation 3 (%)"]

# List of columns for waves dropdown
waves_columns = waves_data.columns[1:6]

# Load wind data
# wind_data = process_wind_data(WIND_DATA_DIR)

# Correctly parse the 'Time and Date' column as datetime
wind_data['datetime'] = pd.to_datetime(wind_data['Time and Date'], errors='coerce')

# Split GPS column into latitude and longitude
wind_data[['latitude', 'longitude']] = wind_data['GPS'].str.split(' ', expand=True).astype(float)

# Heights for wind speed and wind direction
heights = [257, 227, 197, 177, 157, 137, 127, 107, 87, 57, 38]

# Initialize the app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])
server = app.server

# Layout of the dashboard using Bootstrap components
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1('Lidar Sensor Data Monitoring', className='text-center'), width=12)
    ], style={'padding': '20px'}),

    dbc.Row([
        dbc.Col([
            # GPS Geoplot
            dcc.Graph(id='gps-map', style={'height': '400px'},
                      figure=px.scatter_mapbox(wind_data, lat='latitude', lon='longitude',
                                               mapbox_style = "open-street-map",
                                               title='GPS Location', zoom=10).update_layout(
                                                   margin={"r":15,"t":40,"l":15,"b":20},
                                                   paper_bgcolor='rgb(235, 245, 240)', autosize=True).update_traces(
                                                       marker=dict(color='red', size=10))
                                               ),
        ], width=4, style={'padding': '10px'}),

        dbc.Col([
            # Subplots for Temperature and Pressure vs Time
            dcc.Graph(id='temp-pressure-time')
        ], width=8, style={'padding': '10px'})
    ]),

    dbc.Row([
        dbc.Col([
            # Wind Speed and Direction subplots
            dbc.Card([
                dbc.CardBody([
                    dbc.Checklist(
                        id='wind-checklist',
                        options=[{'label': f'{height}m', 'value': height} for height in heights],
                        value=[257],  # Default height selection
                        inline=True
                    ),
                    dcc.Graph(id='wind-subplots')
                ])
            ]),
        ], width=12, style={'padding': '10px'})
    ]),

    # New row for the Wind Speed Histogram with separate checklist
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    # html.H4("Select Heights for Wind Speed Histogram"),
                    dbc.Checklist(
                        id='wind-histogram-checklist',
                        options=[{'label': f'{height}m', 'value': height} for height in heights],
                        value=[257],  # Default height selection
                        inline=True
                    ),
                    dcc.Graph(id='wind-speed-histogram', style={'height': '475px'})
                ])
            ]),
        ], width=12, style={'padding': '10px'})
    ]),

    dbc.Row([
        dbc.Col([
            # Wind Speed Heatmap
            dbc.Card([
                dbc.CardBody([
                    dbc.Checklist(
                        id='wind-speed-checklist',
                        options=[{'label': f'{height}m', 'value': f'Horizontal Wind Speed (m/s) at {height}m'} for height in heights],
                        value=[f'Horizontal Wind Speed (m/s) at {heights[0]}m'],  # Default selection
                        inline=True
                    ),
                    dcc.Graph(id='wind-speed-heatmap', style={'height': '400px'})
                ])
            ]),
        ], width=12, style={'padding': '10px'}),

        dbc.Col([
            # Wind Direction Heatmap
            dbc.Card([
                dbc.CardBody([
                    dbc.Checklist(
                        id='wind-direction-checklist',
                        options=[{'label': f'{height}m', 'value': f'Wind Direction (deg) at {height}m'} for height in heights],
                        value=[f'Wind Direction (deg) at {heights[0]}m'],  # Default selection
                        inline=True
                    ),
                    dcc.Graph(id='wind-direction-heatmap', style={'height': '400px'})
                ])
            ]),
        ], width=12, style={'padding': '10px'})
    ]),

    dbc.Row([
        dbc.Col([
            html.H3("Currents Data"),
            dcc.Dropdown(
                id='currents-yaxis-column',
                options=[{'label': col, 'value': col} for col in currents_columns],
                value=currents_columns[0],  # Default value
                    style={
                    'color': 'black',  # This changes the text color to black
                    'backgroundColor': 'white',  # Optional: Set background color
                    # 'borderColor': 'black',  # Optional: Set border color
                    # 'font-size': '14px',  # Optional: Adjust font size
                    # 'width': '50%'  # Optional: Adjust the width of the dropdown
                },
                clearable=False
            ),
            dcc.Graph(id='currents-time-series-graph', style={'height': '400px'})
        ], width=6, style={'padding': '10px'}),

        dbc.Col([
            html.H3("Waves Data"),
            dcc.Dropdown(
                id='waves-yaxis-column',
                options=[{'label': col, 'value': col} for col in waves_columns],
                value=waves_columns[0],  # Default value
                    style={
                    'color': 'black',  # This changes the text color to black
                    'backgroundColor': 'white',  # Optional: Set background color
                    # 'borderColor': 'black',  # Optional: Set border color
                    # 'font-size': '14px',  # Optional: Adjust font size
                    # 'width': '50%'  # Optional: Adjust the width of the dropdown
                },
                clearable=True
            ),
            dcc.Graph(id='waves-time-series-graph', style={'height': '400px'})
        ], width=6, style={'padding': '10px'})
    ]),

    # Add this row to the layout for the Hs KDE plot
    dbc.Row([
        dbc.Col([
            html.H3("Hs KDE Plot"),
            dcc.Graph(id='hs-kde-plot', style={'height': '400px'})
        ], width=12, style={'padding': '10px'})
    ]),

], fluid=True)

# Callback for updating the Temperature and Pressure vs Time subplot
@app.callback(
    Output('temp-pressure-time', 'figure'),
    Input('temp-pressure-time', 'id')  # Dummy input to trigger the callback
)
def update_temp_pressure_subplot(_):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.02)

    fig.add_trace(
        go.Scatter(x=wind_data['datetime'], y=wind_data['Met Air Temp. (C)'], mode='lines', name='Air Temperature (C)'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=wind_data['datetime'], y=wind_data['Met Pressure (mbar)'], mode='lines', name='Pressure (mbar)'),
        row=2, col=1
    )

    fig.update_layout(height=400, showlegend=False, title='Temperature and Pressure vs Time')
    fig.update_yaxes(title_text='Temperature (C)', row=1, col=1)
    fig.update_yaxes(title_text='Pressure (mbar)', row=2, col=1)
    fig.update_xaxes(title_text='Time', row=2, col=1)
    fig.update_layout(margin={"r":15,"t":40,"l":15,"b":20}, autosize=True)

    return fig

# Callback for updating the Wind Direction and Wind Speed subplots based on checklist selection
@app.callback(
    Output('wind-subplots', 'figure'),
    Input('wind-checklist', 'value')
)
def update_wind_subplots(selected_heights):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    for height in selected_heights:
        wind_direction_col = f'Wind Direction (deg) at {height}m'
        fig.add_trace(
            go.Scatter(x=wind_data['datetime'], y=wind_data[wind_direction_col], mode='lines', name=f'WD {height}m'),
            row=1, col=1
        )
    for height in selected_heights:
        wind_speed_col = f'Horizontal Wind Speed (m/s) at {height}m'
        fig.add_trace(
            go.Scatter(x=wind_data['datetime'], y=wind_data[wind_speed_col], mode='lines', name=f'WS {height}m'),
            row=2, col=1
        )

    fig.update_layout(height=400, title='Wind Direction and Speed at Selected Heights')
    fig.update_yaxes(title_text='Wind Direction (deg)', row=1, col=1)
    fig.update_yaxes(title_text='Wind Speed (m/s)', row=2, col=1)
    fig.update_xaxes(title_text='Time', row=2, col=1)
    fig.update_layout(margin={"r":10,"t":30,"l":10,"b":5}, autosize=True)

    return fig

# Callback for updating the Wind Speed histogram with KDE curve based on a separate checklist
@app.callback(
    Output('wind-speed-histogram', 'figure'),
    Input('wind-histogram-checklist', 'value')
)
def update_wind_speed_histogram(selected_heights):
    hist_data = []
    group_labels = []

    for height in selected_heights:
        wind_speed_col = f'Horizontal Wind Speed (m/s) at {height}m'
        hist_data.append(wind_data[wind_speed_col].dropna())  # Remove NaN values
        group_labels.append(f'{height}m')

    fig = ff.create_distplot(hist_data, group_labels, bin_size=0.3, show_hist=True, show_rug=False)
    fig.update_layout(title='Wind Speed Histogram with KDE Curve',
                      xaxis_title='WS(m/s)',
                      yaxis_title='Density')
    fig.update_layout(margin={"r":10,"t":30,"l":10,"b":5}, autosize=True)
    return fig


# Function to extract the height as an integer from the column name
def extract_height(column_name):
    match = re.search(r'(\d+)m', column_name)
    if match:
        return int(match.group(1))
    return float('inf')  # If no number found, put at the end


# Callback for updating the Wind Speed Heatmap
@app.callback(
    Output('wind-speed-heatmap', 'figure'),
    [Input('wind-speed-checklist', 'value')]
)
def update_wind_speed_heatmap(selected_columns):
    if not selected_columns:
        return go.Figure()
    
    sorted_columns = sorted(selected_columns, key=extract_height, reverse=True)

    wind_speed_data = wind_data[['datetime'] + sorted_columns]
    wind_speed_data.set_index('datetime', inplace=True)  # Set Datetime as index
    
    # Create the heatmap with Datetime as X-axis labels
    fig = px.imshow(wind_speed_data.T, aspect='auto',
                    labels={'x': 'Datetime', 'y': 'Wind Speed Height'},
                    x=wind_speed_data.index,
                    color_continuous_scale='RdBu_r')
    
    fig.update_layout(title='Wind Speed Heatmap')
    return fig


# Callback for updating the Wind Direction Heatmap
@app.callback(
    Output('wind-direction-heatmap', 'figure'),
    [Input('wind-direction-checklist', 'value')]
)
def update_wind_direction_heatmap(selected_columns):
    if not selected_columns:
        return go.Figure()

    sorted_columns = sorted(selected_columns, key=extract_height, reverse=True)

    wind_direction_data = wind_data[['datetime'] + sorted_columns]
    wind_direction_data.set_index('datetime', inplace=True)  # Set Datetime as index

    # Create the heatmap with Datetime as X-axis labels
    fig = px.imshow(wind_direction_data.T, aspect='auto',
                    labels={'x': 'Datetime', 'y': 'Wind Direction Height'},
                    x=wind_direction_data.index,
                    color_continuous_scale='RdBu_r')

    fig.update_layout(title='Wind Direction Heatmap')
    return fig


# Callback for updating the Currents Time Series Graph
@app.callback(
    Output('currents-time-series-graph', 'figure'),
    Input('currents-yaxis-column', 'value')
)
def update_currents_graph(selected_column):
    # Calculate the last two hours range
    last_two_hours_start = currents_data['Datetime'].max() - pd.Timedelta(hours=24)
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=currents_data['Datetime'], y=currents_data[selected_column], mode='lines+markers', name=selected_column)
    )
    # Setting the xaxis range to the last two hours but showing all data
    fig.update_layout(
        title=f'{selected_column} vs Time (Currents)',
        xaxis_title='Datetime',
        yaxis_title=selected_column,
        xaxis_range=[last_two_hours_start, currents_data['Datetime'].max()]  # Default zoom to last 2 hours
    )
    fig.update_layout(margin={"r":15,"t":40,"l":15,"b":10}, autosize=True)
    return fig

# Callback for updating the Waves Time Series Graph
@app.callback(
    Output('waves-time-series-graph', 'figure'),
    Input('waves-yaxis-column', 'value')
)
def update_waves_graph(selected_column):
    # Calculate the last two hours range
    last_two_hours_start = waves_data['Datetime'].max() - pd.Timedelta(hours=24)
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=waves_data['Datetime'], y=waves_data[selected_column], mode='lines+markers', name=selected_column)
    )
    # Setting the xaxis range to the last two hours but showing all data
    fig.update_layout(
        title=f'{selected_column} vs Time (Waves)',
        xaxis_title='Datetime',
        yaxis_title=selected_column,
        xaxis_range=[last_two_hours_start, waves_data['Datetime'].max()]  # Default zoom to last 2 hours
    )
    fig.update_layout(margin={"r":15,"t":40,"l":15,"b":10}, autosize=True)
    return fig

# Callback for updating the Hs KDE plot
@app.callback(
    Output('hs-kde-plot', 'figure'),
    Input('hs-kde-plot', 'id')  # Dummy input to trigger the callback
)
def update_hs_kde(_):
    # Drop NaN values in Hs column
    hs_data = waves_data['Hs'].dropna()
    hs_data = pd.to_numeric(hs_data, errors='coerce')

    # Create KDE plot using plotly figure factory
    fig = ff.create_distplot([hs_data], group_labels=['Hs'], bin_size=0.2, show_hist=True, show_rug=False)
    
    # Update layout of the KDE plot
    fig.update_layout(
        title='KDE Plot for Hs',
        xaxis_title='Hs (m)',
        yaxis_title='Density',
        margin={"r":10, "t":30, "l":10, "b":20}
    )
    return fig


# Start the app
if __name__ == "__main__":
    app.run_server(debug=False)
