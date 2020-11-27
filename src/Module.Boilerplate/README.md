# Development
## Setup and basic commands 
1. Install Python 3.8 or greater  
2. Install Potery - https://python-poetry.org/docs/#installation  
3. `poetry install` - install dependencies 
4. `poetry add {name}` - add new dependency

## Running 
### Local 
`python main.py` or `uvicorn main:app`
### Docker 
`docker run --rm -it -p 8000:8000 bolier uvicorn main:app --host 0.0.0.0 --port 8000`