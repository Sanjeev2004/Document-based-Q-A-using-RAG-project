# Quick Start Guide

## 1. Activate Virtual Environment

```powershell
.\venv\Scripts\activate
```

## 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

## 3. Run Health Checks (Recommended)

```powershell
python health_check.py
```

If you only want local checks:

```powershell
python health_check.py --skip-llm
```

## 4. Ingest a Document

Command line option:

```powershell
python src\ingestion.py "data\SANJEEV_KUMAR_2K22CO408.pdf"
```

Or ingest from the Streamlit UI.

## 5. Run the App

```powershell
streamlit run app.py
```

## 6. Optional: Run Tests

```powershell
pytest -q
```

## Troubleshooting

- If dependencies fail to install, confirm the virtual environment is active.
- If health check fails on model access, verify `HUGGINGFACE_API_KEY` and model permissions.
- If ingestion fails, verify the PDF path and that text is extractable.
- If browser does not open, manually visit `http://localhost:8501`.
