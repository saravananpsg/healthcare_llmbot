import os
import requests

# Importing required functionalities
from PyPDF2 import PdfReader

infile_path = "urls_list.txt"
with open(infile_path, 'r') as infile:
    urls_data=infile.readlines()

def download_pdf(url, file_Path):
    response = requests.get(url) 
    if response.status_code == 200:
        with open(file_Path, 'wb') as file:
            file.write(response.content)
        print('File downloaded successfully')
    else:
        print('Failed to download file')

# Extract text from a given PDF file
def extract_pdf_text(file_path):
    pdf_file = PdfReader(file_path)
    text_data = ''
    for pg in pdf_file.pages:
        text_data += pg.extract_text()
    return text_data