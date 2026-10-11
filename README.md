# CodeAlpha_ProjectName

This project trains a K-Nearest Neighbors (KNN) classifier on the Iris flower dataset and evaluates its performance.

## Project structure

- `main.py` — training and evaluation script
- `requirements.txt` — project dependencies
- `README.md` — usage instructions

## Run the project

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Notes

The script first tries to load `Iris.csv` from the project root. If that file does not exist, it falls back to scikit-learn's built-in Iris dataset so the project still runs without extra setup.
