# AI-Project-Road-classifire-
Road Classifier for Project
Uses CNN model to be trained

## Run locally
```
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000/

## Deploy on Render
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app --workers 1 --threads 1 --timeout 180`
- Health check path: `/health`
