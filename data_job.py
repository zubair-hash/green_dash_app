import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from pathlib import Path
import paramiko
from dotenv import load_dotenv
import stat
import glob

load_dotenv()

app = FastAPI()

# Define SFTP credentials and directories
SFTP_HOST = os.getenv('SFTP_HOST')
SFTP_USERNAME = os.getenv('SFTP_USERNAME')
SFTP_PASSWORD = os.getenv('SFTP_PASSWORD')
REMOTE_BASE_PATH = os.getenv('REMOTE_FILE_PATH')

LOCAL_BASE_DIRECTORY = Path("./data")
CURRENT_DATA_DIR = LOCAL_BASE_DIRECTORY / "Current" / "XX03.nmea"
WAVE_DATA_DIR = LOCAL_BASE_DIRECTORY / "Wave" / "COM3_2024_09_21.txt"
WIND_DATA_DIR = LOCAL_BASE_DIRECTORY / "Lidar"

# SFTP functions
def get_remote_file_info(sftp, file_path):
    file_attributes = sftp.stat(file_path)
    return file_attributes.st_mtime

def get_local_file_info(local_file_path):
    if local_file_path.exists():
        return local_file_path.stat().st_mtime
    return None

def is_file_updated(local_mtime, remote_mtime):
    return local_mtime is None or remote_mtime > local_mtime

def download_file(sftp, remote_file, local_file):
    sftp.get(remote_file, str(local_file))
    print(f"Downloaded {remote_file} to {local_file}")

def is_directory(sftp, path):
    try:
        return stat.S_ISDIR(sftp.stat(path).st_mode)
    except IOError:
        return False

def mirror_directory_structure(sftp, remote_dir, local_dir):
    if not local_dir.exists():
        local_dir.mkdir(parents=True)
        print(f"Created local directory: {local_dir}")
    for item in sftp.listdir(remote_dir):
        remote_item_path = f"{remote_dir}/{item}"
        local_item_path = local_dir / item
        if is_directory(sftp, remote_item_path):
            mirror_directory_structure(sftp, remote_item_path, local_item_path)

def check_subdirectory(sftp, remote_dir, local_dir):
    directory_contents = sftp.listdir(remote_dir)
    for item in directory_contents:
        remote_file = f"{remote_dir}/{item}"
        local_file = local_dir / item
        if is_directory(sftp, remote_file):
            check_subdirectory(sftp, remote_file, local_file)
        else:
            remote_mtime = get_remote_file_info(sftp, remote_file)
            local_mtime = get_local_file_info(local_file)
            if is_file_updated(local_mtime, remote_mtime):
                download_file(sftp, remote_file, local_file)

def fetch_data_from_sftp():
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(SFTP_HOST, username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = ssh.open_sftp()
        sftp.chdir(REMOTE_BASE_PATH)
        mirror_directory_structure(sftp, REMOTE_BASE_PATH, LOCAL_BASE_DIRECTORY)
        check_subdirectory(sftp, REMOTE_BASE_PATH, LOCAL_BASE_DIRECTORY)
        sftp.close()
        ssh.close()
    except Exception as e:
        print(f"An error occurred: {e}")

# Processing functions
def process_currents_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = [line.strip().split(',') for line in lines if line.startswith('$PNORC')]
    df = pd.DataFrame(data)
    df.replace('', pd.NA, inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    column_names = [
        "Identifier", "Date", "Time", "Cell number", "Velocity 1 (m/s)", "Velocity 2 (m/s)", 
        "Velocity 3 (m/s)", "Speed (m/s)", "Direction (°)", "Amplitude units", 
        "Correlation 1 (%)", "Correlation 2 (%)", "Correlation 3 (%)", "Checksum (hex)"
    ]
    if df.shape[1] <= len(column_names):
        df.columns = column_names[:df.shape[1]]
    
    def convert_to_datetime(row):
        date_str = str(row['Date'])[-5:]
        month, day, year = date_str[0].zfill(2), date_str[1:3], "20" + date_str[3:]
        time_str = str(row['Time']).zfill(6)
        datetime_str = f"{year}-{month}-{day} {time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
        return pd.Timestamp(datetime_str)

    if 'Date' in df.columns and 'Time' in df.columns:
        df['Datetime'] = df.apply(convert_to_datetime, axis=1)
    return df

def process_wave_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = [line.strip().split(',') for line in lines if line.strip()]
    column_names = ['identifier','NMEA', 'CompassHeading','Hs','DominantPeriod','DominantPeriodFW'] + \
                   [f'parameter{i}' for i in range(6, 22)] + ['Datetime', 'parameter22']
    df = pd.DataFrame(data, columns=column_names)
    return df

def process_wind_data(wind_data_dir):
    all_files = glob.glob(str(wind_data_dir / 'Wind10_829@Y2024_M09_D*.ZPH.csv'))
    wind_data = pd.concat((pd.read_csv(f, skiprows=1) for f in all_files), ignore_index=True)
    wind_data.replace(9999, np.nan, inplace=True)
    wind_data.ffill(inplace=True)
    wind_data.bfill(inplace=True)
    return wind_data

# API endpoints
@app.get("/data/currents")
async def get_currents_data():
    fetch_data_from_sftp()
    df = process_currents_data(CURRENT_DATA_DIR)
    return df.to_dict(orient="records")

@app.get("/data/waves")
async def get_waves_data():
    fetch_data_from_sftp()
    df = process_wave_data(WAVE_DATA_DIR)
    return df.to_dict(orient="records")

@app.get("/data/wind")
async def get_wind_data():
    fetch_data_from_sftp()
    df = process_wind_data(WIND_DATA_DIR)
    return df.to_dict(orient="records")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
