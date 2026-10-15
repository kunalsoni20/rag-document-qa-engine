import os
import secrets
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from rag_engine import RAGEngine

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))

engine = RAGEngine(
    persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", str(BASE_DIR / "chroma_db")),
    upload_directory=UPLOAD_DIR,
)


def _provider_config():
    """Request headers override .env values, which keeps keys out of the browser URL."""
    return {
        "openai_api_key": request.headers.get("X-OpenAI-Key") or os.getenv("OPENAI_API_KEY"),
        "groq_api_key": request.headers.get("X-Groq-Key") or os.getenv("GROQ_API_KEY"),
        "gemini_api_key": request.headers.get("X-Gemini-Key") or os.getenv("GEMINI_API_KEY"),
        "model": request.headers.get("X-LLM-Model") or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/upload")
def upload():
    files = request.files.getlist("files")
    if not files or all(not file.filename for file in files):
        return jsonify({"error": "Select at least one PDF or TXT file."}), 400

    config = _provider_config()
    results = []
    for file in files:
        filename = secure_filename(file.filename or "")
        extension = Path(filename).suffix.lower()
        if not filename or extension not in {".pdf", ".txt"}:
            results.append({"name": filename or "unknown", "status": "error", "message": "Only PDF and TXT files are supported."})
            continue

        stored_path = UPLOAD_DIR / filename
        file.save(stored_path)
        try:
            result = engine.ingest_file(stored_path, config)
            results.append({"name": filename, "status": "indexed", **result})
        except Exception as exc:
            stored_path.unlink(missing_ok=True)
            results.append({"name": filename, "status": "error", "message": str(exc)})

    status_code = 200 if any(item["status"] == "indexed" for item in results) else 400
    return jsonify({"files": results, "documents": engine.list_documents()}), status_code


@app.post("/ask")
def ask():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Ask a question to search your documents."}), 400

    try:
        response = engine.ask(question, payload.get("history", []), _provider_config())
        return jsonify(response)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Question answering failed")
        return jsonify({"error": f"Unable to answer the question: {exc}"}), 500


@app.post("/clear")
def clear():
    try:
        engine.clear()
        return jsonify({"message": "Vector index and uploaded files cleared.", "documents": []})
    except Exception as exc:
        app.logger.exception("Index clearing failed")
        return jsonify({"error": f"Unable to clear the index: {exc}"}), 500


@app.delete("/documents/<path:filename>")
def delete_document(filename):
    try:
        result = engine.delete_document(filename)
        return jsonify({"message": f"{filename} cleared.", "deleted": result, "documents": engine.list_documents()})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        app.logger.exception("Document deletion failed")
        return jsonify({"error": f"Unable to clear {filename}: {exc}"}), 500


@app.get("/documents")
def documents():
    return jsonify({"documents": engine.list_documents()})


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Upload too large. Increase MAX_UPLOAD_MB if needed."}), 413


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
