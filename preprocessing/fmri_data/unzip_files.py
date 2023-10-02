import os
import gzip
from utils.paths import *


# Specify the folder containing the .gz files
# folder = 'your_folder_path'

def unzip_participant_derivative(participant):
    subject_path = os.path.join(data_path,'derivatives',f"{participant}", 'func')
    folder = subject_path
    # Loop through all files in the folder
    for filename in os.listdir(folder):
        if filename.endswith('.gz'):
            # Generate the full path to the .gz file
            file_path = os.path.join(folder, filename)

            # Extract the name without the '.gz' extension
            output_filename = os.path.splitext(filename)[0]

            # Create the output file path
            output_path = os.path.join(folder, output_filename)

            # Open the .gz file for reading and the output file for writing
            with gzip.open(file_path, 'rb') as gz_file, open(output_path, 'wb') as output_file:
                # Read and write data in chunks
                chunk_size = 1024 * 1024  # 1 MB chunks (adjust as needed)
                while True:
                    chunk = gz_file.read(chunk_size)
                    if not chunk:
                        break
                    output_file.write(chunk)

    # Print a message indicating that the extraction is complete
    print("All .gz files have been extracted.")


participant = 'sub-EN057'
unzip_participant_derivative(participant)