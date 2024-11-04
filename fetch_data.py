from pathlib import Path
import paramiko
import os
import stat
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define SFTP credentials and connection parameters from environment variables
sftp_host = os.getenv('SFTP_HOST')
sftp_username = os.getenv('SFTP_USERNAME')
sftp_password = os.getenv('SFTP_PASSWORD')
remote_base_path = os.getenv('REMOTE_FILE_PATH')

# Define the local directory with pathlib for persistent storage
local_base_directory = Path("/data")

def get_remote_file_info(sftp, file_path):
    """Fetch the remote file's modification time."""
    file_attributes = sftp.stat(file_path)
    return file_attributes.st_mtime

def get_local_file_info(local_file_path):
    """Fetch the local file's modification time if it exists."""
    if os.path.exists(local_file_path):
        return os.path.getmtime(local_file_path)
    return None

def is_file_updated(local_mtime, remote_mtime):
    """Compare local and remote modification times."""
    if local_mtime is None:
        return True  # Local file doesn't exist, so consider it "outdated"
    return remote_mtime > local_mtime

def download_file(sftp, remote_file, local_file):
    """Download the file from the remote server."""
    sftp.get(remote_file, local_file)
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
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
            print(f"Created local directory: {local_dir}")
        
        for item in sftp.listdir(remote_dir):
            remote_item_path = f"{remote_dir}/{item}"
            local_item_path = os.path.join(local_dir, item)

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
            local_file = os.path.join(local_dir, item)

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
