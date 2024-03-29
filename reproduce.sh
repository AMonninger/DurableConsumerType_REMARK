#!/bin/bash
# install the versioned required packages
python3 -m pip install --quiet -r requirements.txt

# navigate to code/ and execute the python file to create figures
cd ./code/python
jupyter nbconvert --to python DurableModel_Notebook.ipynb
ipython DurableModel_Notebook.py