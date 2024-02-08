# streamlit_app.py

import streamlit as st
from st_files_connection import FilesConnection
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/linkding/Downloads/Flux.json'


#Create connection object and retrieve file contents.
#Specify input format is a csv and to cache the result for 600 seconds.
conn = st.connection('gcs', type=FilesConnection)
    
# Open and read the local file
def create_file_in_folder(file,file_format):
    with open('Data/{}.{}'.format(file,file_format), 'r') as local_file:
        file_content = local_file.read()
        
    # Write the content to the server
    with conn.open('flux-storage/linkddd/{}.{}'.format(file,file_format), 'w') as server_file:
        server_file.write(file_content)
        
create_file_in_folder('Time','csv')
create_file_in_folder('Project','csv')
create_file_in_folder('combined_data','csv')
create_file_in_folder('TODO','csv')
create_file_in_folder('time_category','json')