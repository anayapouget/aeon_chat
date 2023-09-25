# aeon_chat
This repo contains the code needed to launch a locally hosted chatbot focused on question answering over Project Aeon, specifically the aeon_mecha and aeon_analysis repositories.

## Setup
1. Create an aeon_chat conda environment using the repo's environment.yml file:
```bash
conda env create -f environment.yml
```
2. Add your chatgpt API key to the constants.py file:
```python
APIKEY = "YOUR_API_KEY"
```
3. (Optional) 
    1. Clone the aeon_mecha and aeon_analyis repos and copy their paths over to ingest_data.ipynb:
    ```python
    aeon_analysis = "path/to/aeon_analysis"
    aeon_mecha = "path/to/aeon_mecha"
    ```
    2. Run the cells in the ingest_data notebook to update the Chroma vectorstore which Aeon Chat will use to answer questions.

## Launch
1. Open the command line and navigate to the aeon_chat repo.
2. Activate your aeon_chat environment:
```bash
conda activate aeon_chat
```
3. Run:
```bash
python app.py
```
