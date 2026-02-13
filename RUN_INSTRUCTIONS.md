# Quick Start Guide

## Step-by-Step Instructions

### 1. Activate Virtual Environment
```powershell
.\venv\Scripts\activate
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Ingest Your Document
```powershell
python ingest.py "data\SANJEEV_KUMAR_2K22CO408.pdf"
```

### 4. Run the Application
```powershell
streamlit run app.py
```

### 5. Open in Browser
The app will automatically open at: `http://localhost:8501`

## Troubleshooting

- **If dependencies fail to install**: Make sure your venv is activated
- **If Pinecone errors**: Check your `.env` file has correct API keys
- **If document ingestion fails**: Verify the PDF file path is correct
- **If browser doesn't open**: Manually navigate to `http://localhost:8501`

## Example Questions to Try

After ingesting your document, try asking:
- "What is this document about?"
- "Summarize the key information"
- "What are the main topics discussed?"
- "What skills or qualifications are mentioned?"
