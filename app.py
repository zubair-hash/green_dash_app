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
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Define SFTP credentials and connection parameters from environment variables
sftp_host = os.getenv('SFTP_HOST')
sftp_username = os.getenv('SFTP_USERNAME')
sftp_password = os.getenv('SFTP_PASSWORD')
remote_base_path = os.getenv('REMOTE_FILE_PATH')  # Base path for the main directory

# Define the local directory where you want to save the files (ephemeral on Heroku)
local_base_directory = Path("./data")

def get_remote_file_info(sftp, file_path):
    """Fetch the remote file's modification time."""
    file_attributes = sftp.stat(file_path)
    return file_attributes.st_mtime

def get_local_file_info(local_file_path):
    """Fetch the local file's modification time if it exists."""
    if local_file_path.exists():
        return local_file_path.stat().st_mtime
    return None

def is_file_updated(local_mtime, remote_mtime):
    """Compare local and remote modification times."""
    if local_mtime is None:
        return True  # Local file doesn't exist, so consider it "outdated"
    return remote_mtime > local_mtime

def download_file(sftp, remote_file, local_file):
    """Download the file from the remote server."""
    sftp.get(remote_file, str(local_file))
    print(f"Downloaded {remote_file} to {local_file}")

def is_directory(sftp, path):
    """Check if the path is a directory on the remote server."""
    try:
        return stat.S_ISDIR(sftp.stat(path).st_mode)
    except IOError:
        return False

def mirror_directory_structure(sftp, remote_dir, local_dir):
    """Mirror the remote directory structure locally."""
    try:
        if not local_dir.exists():
            local_dir.mkdir(parents=True)
            print(f"Created local directory: {local_dir}")
        
        for item in sftp.listdir(remote_dir):
            remote_item_path = f"{remote_dir}/{item}"
            local_item_path = local_dir / item

            # If the item is a directory, recurse into it
            if is_directory(sftp, remote_item_path):
                mirror_directory_structure(sftp, remote_item_path, local_item_path)
            else:
                # Skip for now, files will be downloaded in a separate process
                pass

    except Exception as e:
        print(f"An error occurred while mirroring directory structure: {e}")

def check_subdirectory(sftp, remote_dir, local_dir):
    """Check and download updated files from a specific subdirectory."""
    directory_contents = sftp.listdir(remote_dir)

    if len(directory_contents) == 0:
        print(f"No Data Available in {remote_dir}.")
    else:
        print(f"Checking for updated files in {local_dir}...")

        for item in directory_contents:
            remote_file = f"{remote_dir}/{item}"
            local_file = local_dir / item

            if is_directory(sftp, remote_file):
                check_subdirectory(sftp, remote_file, local_file)
            else:
                remote_mtime = get_remote_file_info(sftp, remote_file)
                local_mtime = get_local_file_info(local_file)

                if is_file_updated(local_mtime, remote_mtime):
                    print(f"Newer version found for {item}. Downloading...")
                    download_file(sftp, remote_file, local_file)
                else:
                    print(f"{item} is already up to date.")

def process_data():
    """Main function to check and download updated data."""
    try:
        # Initialize SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the SFTP server
        ssh.connect(sftp_host, username=sftp_username, password=sftp_password)

        # Open an SFTP session
        sftp = ssh.open_sftp()

        print("Connected successfully to the SFTP server.")

        # Change to the base remote directory
        sftp.chdir(remote_base_path)

        # Mirror the entire directory structure before downloading any files
        mirror_directory_structure(sftp, remote_base_path, local_base_directory)

        # Now, proceed to check and download updated files from each subdirectory
        check_subdirectory(sftp, remote_base_path, local_base_directory)

        # Close the SFTP session and SSH connection
        sftp.close()
        ssh.close()

    except Exception as e:
        print(f"An error occurred: {e}")

# Call the main process function to perform the data check and download
process_data()

CURRENT_DATA_DIR = local_base_directory / "Current/XX03.nmea"
WAVE_DATA_DIR = local_base_directory / "Wave/COM3_2024_09_21.txt"
WIND_DATA_DIR = local_base_directory / "Lidar"

def process_currents_data(file_path):
    # Read the NMEA file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Filter for lines that start with $PNORC and split by commas
    data = [line.strip().split(',') for line in lines if line.startswith('$PNORC')]

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Drop empty columns (columns with all NaN or empty values)
    df.replace('', pd.NA, inplace=True)  # Replace empty strings with NaN
    df.dropna(axis=1, how='all', inplace=True)

    column_names = [
        "Identifier", "Date", "Time", "Cell number", "Velocity 1 (m/s)", "Velocity 2 (m/s)", 
        "Velocity 3 (m/s)", "Speed (m/s)", "Direction (°)", "Amplitude units", 
        "Correlation 1 (%)", "Correlation 2 (%)", "Correlation 3 (%)", "Checksum (hex)"
    ]

    if df.shape[1] <= len(column_names):
        df.columns = column_names[:df.shape[1]]
    else:
        print("DataFrame has unexpected number of columns:", df.shape[1])
        print(df.head())
        return

    def convert_to_datetime(row):
        date_str = str(row['Date'])
        if len(date_str) > 5:
            date_str = date_str[-5:]

        month = date_str[0].zfill(2)
        day = date_str[1:3]
        year = "20" + date_str[3:]
        
        time_str = str(row['Time']).zfill(6)
        hours = time_str[:2]
        minutes = time_str[2:4]
        seconds = time_str[4:]

        datetime_str = f"{year}-{month}-{day} {hours}:{minutes}:{seconds}"
        return pd.Timestamp(datetime_str)

    if 'Date' in df.columns and 'Time' in df.columns:
        df['Datetime'] = df.apply(convert_to_datetime, axis=1)
    else:
        print("Missing 'Date' or 'Time' columns in the DataFrame!")
        return

    return df

def process_wave_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data = [line.strip().split(',') for line in lines if line.strip()]

    df = pd.DataFrame(data)

    column_names = ['identifier', 'NMEA', 'CompassHeading', 'Hs', 'DominantPeriod', 'DominantPeriodFW'] + \
                   [f'parameter{i}' for i in range(6, 22)] + ['Datetime', 'parameter22']
    df.columns = column_names
    
    return df

def process_wind_data(wind_data_dir):
    csv_pattern = str(wind_data_dir / 'Wind10_829@Y2024_M09_D*.ZPH.csv')
    
    all_files = glob.glob(csv_pattern)
    
    if not all_files:
        print("No wind data files found!")
        return
    
    wind_data = pd.concat((pd.read_csv(f, skiprows=1) for f in all_files), ignore_index=True)

    wind_data.replace(9999, np.nan, inplace=True)
    wind_data.ffill(inplace=True)
    wind_data.bfill(inplace=True)

    return wind_data

# Load currents data
currents_data = process_currents_data(CURRENT_DATA_DIR) # comment if you are loading fron the folder
currents_data['Datetime'] = pd.to_datetime(currents_data['Datetime'])

# Load waves data
waves_data = process_wave_data(WAVE_DATA_DIR) # comment if you are loading fron the folder
waves_data['Datetime'] = pd.to_datetime(waves_data['Datetime'])

# List of columns for currents dropdown
currents_columns = ["Velocity 1 (m/s)", "Velocity 2 (m/s)", "Velocity 3 (m/s)", "Speed (m/s)", "Direction (°)",
                    "Amplitude units", "Correlation 1 (%)", "Correlation 2 (%)", "Correlation 3 (%)"]

# List of columns for waves dropdown
waves_columns = waves_data.columns[1:6]

# Load wind data
wind_data = process_wind_data(WIND_DATA_DIR)

# Correctly parse the 'Time and Date' column into datetime format
wind_data['Datetime'] = pd.to_datetime(wind_data['Time and Date'], format='%d-%m-%Y %H:%M:%S', errors='coerce')

# Remove duplicate rows
wind_data.drop_duplicates(inplace=True)

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
if __name__ == '__main__':
    app.run_server(debug=True, port=10000)