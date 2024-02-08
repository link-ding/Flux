# streamlit_app.py

import streamlit as st
from st_files_connection import FilesConnection
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/linkding/Downloads/Flux.json'


#Create connection object and retrieve file contents.
#Specify input format is a csv and to cache the result for 600 seconds.
conn = st.connection('gcs', type=FilesConnection)
conn.read('flux-storage')