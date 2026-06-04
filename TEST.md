# Run and Test Guide

All commands below are intended to be run from the project root.

## 1. Prepare the environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place images referenced by `data/dataset.csv` under `data/images/`, or update
the CSV paths to point to your own images.

## 2. Build or rebuild both indexes

```powershell
python build_index.py
```

## 3. Run the API

```powershell
uvicorn app:app --reload
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## 4. Test API endpoints

Run these commands in a second PowerShell terminal while the API is running.

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"
```

```powershell
$body = @{ query = "worker without helmet near excavator"; top_k = 5 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/search/caption" -ContentType "application/json" -Body $body
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/search/image" -ContentType "application/json" -Body $body
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/search/hybrid" -ContentType "application/json" -Body $body
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/rag/answer" -ContentType "application/json" -Body $body
```

## 5. Optional static checks

```powershell
python -m compileall app.py build_index.py config.py rag_answer.py retriever.py
```
