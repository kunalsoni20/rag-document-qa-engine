$ErrorActionPreference = "Stop"

Write-Host "Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements-windows.txt

Write-Host "Installing the available Windows Chroma native wheel..."
python -m pip install chroma-hnswlib==0.7.5

Write-Host "Installing ChromaDB without re-resolving its unavailable native dependency..."
python -m pip install --no-deps chromadb==0.6.3
python -m pip install --no-deps langchain-chroma==0.2.1

Write-Host "Verifying application imports..."
python -c "import dotenv, overrides, chromadb, langchain_chroma; print('Dependencies ready.')"
Write-Host "Setup complete. Start the app with: python app.py"
