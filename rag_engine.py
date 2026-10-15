import os
import re
import shutil
from pathlib import Path
from typing import Any

import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


class RAGEngine:
    """Local persistent RAG pipeline with optional hosted generation providers."""

    def __init__(self, persist_directory: str, upload_directory: Path):
        self.persist_directory = Path(persist_directory)
        self.upload_directory = Path(upload_directory)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""]
        )
        self._vectorstore = None
        self._embedding_signature = None

    def _embeddings(self, config: dict[str, str | None]) -> Embeddings:
        openai_key = config.get("openai_api_key")
        signature = "openai" if openai_key else "huggingface"
        if self._embedding_signature == signature and self._vectorstore is not None:
            return self._vectorstore._embedding_function

        if openai_key:
            from langchain_openai import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)
        else:
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name=os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                model_kwargs={"device": os.getenv("HF_DEVICE", "cpu")},
                encode_kwargs={"normalize_embeddings": True},
            )
        return embeddings

    def _store(self, config: dict[str, str | None]) -> Chroma:
        embeddings = self._embeddings(config)
        signature = "openai" if config.get("openai_api_key") else "huggingface"
        if self._vectorstore is None or self._embedding_signature != signature:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._vectorstore = Chroma(
                collection_name=f"local_document_chunks_{signature}",
                embedding_function=embeddings,
                persist_directory=str(self.persist_directory),
            )
            self._embedding_signature = signature
        return self._vectorstore

    def _extract_documents(self, path: Path) -> list[Document]:
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            documents = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    documents.append(Document(page_content=text, metadata={"page": page_number}))
            return documents
        return [Document(page_content=path.read_text(encoding="utf-8", errors="replace"), metadata={"page": 1})]

    def ingest_file(self, path: Path, config: dict[str, str | None]) -> dict[str, Any]:
        source_documents = self._extract_documents(path)
        if not source_documents:
            raise ValueError(f"No readable text found in {path.name}.")

        chunks = self.text_splitter.split_documents(source_documents)
        for index, chunk in enumerate(chunks):
            chunk.metadata.update({"source": path.name, "chunk": index + 1})
        ids = [f"{path.name}:{index}" for index in range(len(chunks))]
        self._store(config).add_documents(chunks, ids=ids)
        return {"chunks": len(chunks), "pages": len(source_documents)}

    def _retrieve(self, question: str, config: dict[str, str | None], k: int = 5) -> list[dict[str, Any]]:
        store = self._store(config)
        if store._collection.count() == 0:
            raise ValueError("Upload and index a document before asking a question.")
        source_filter = None
        filenames = {path.name.lower(): path.name for path in self.upload_directory.iterdir() if path.is_file()}
        for filename_lower, filename in filenames.items():
            if filename_lower in question.lower():
                source_filter = {"source": filename}
                break
        search_kwargs = {"filter": source_filter} if source_filter else {}
        matches = store.similarity_search_with_score(question, k=max(k, 30), **search_kwargs)
        query_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
        exact_phrase = question.lower().strip(" ?.")
        practical_match = re.search(r"\bpractical\s*[-:]?\s*0?(\d+)\b", question.lower())
        requested_practical = practical_match.group(1) if practical_match else None

        def lexical_score(document: Document) -> tuple[int, int, int]:
            content = document.page_content.lower()
            content_terms = set(re.findall(r"[a-z0-9]+", content))
            practical_score = 0
            if requested_practical:
                practical_patterns = (
                    rf"\bpractical\s*[-:]?\s*0?{requested_practical}\b",
                    rf"\bpractical\s*[-:]?\s*0?{requested_practical}\s*aim\b",
                )
                practical_score = 100 if any(re.search(pattern, content) for pattern in practical_patterns) else 0
            return (
                practical_score,
                int(exact_phrase in content),
                len(query_terms.intersection(content_terms)),
            )

        ranked_matches = sorted(
            matches,
            key=lambda pair: (
                *lexical_score(pair[0]),
                -float(pair[1]),
            ),
            reverse=True,
        )[:k]
        distances = [float(distance) for _, distance in ranked_matches]
        minimum_distance = min(distances)
        maximum_distance = max(distances)
        citations = []
        for document, relevance in ranked_matches:
            if maximum_distance == minimum_distance:
                similarity = 1.0
            else:
                similarity = (maximum_distance - float(relevance)) / (maximum_distance - minimum_distance)
            citations.append(
                {
                    "source": document.metadata.get("source", "Unknown"),
                    "page": document.metadata.get("page", 1),
                    "chunk": document.metadata.get("chunk"),
                    "text": document.page_content,
                    "similarity": round(similarity, 4),
                }
            )
        return citations

    def _context(self, citations: list[dict[str, Any]]) -> str:
        return "\n\n".join(
            f"[Source: {item['source']}, page {item['page']}]\n{item['text']}" for item in citations
        )

    def _hosted_answer(self, question: str, context: str, history: list[dict[str, str]], config: dict[str, str | None]) -> tuple[str, str]:
        system = (
            "You are a precise document question-answering assistant. Answer only from the supplied context. "
            "If the context does not contain the answer, say so clearly. Never invent facts. "
            "Use concise Markdown and mention relevant source names/pages inline when useful."
        )
        messages = [{"role": "system", "content": system}]
        for item in history[-6:]:
            if item.get("role") in {"user", "assistant"} and item.get("content"):
                messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

        if config.get("openai_api_key"):
            from langchain_openai import ChatOpenAI

            result = ChatOpenAI(model=config.get("model") or "gpt-4o-mini", api_key=config["openai_api_key"], temperature=0).invoke(messages)
            return result.content, "OpenAI"

        if config.get("groq_api_key"):
            response = requests.post(
                os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions"),
                headers={"Authorization": f"Bearer {config['groq_api_key'].strip()}", "Content-Type": "application/json"},
                json={"model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), "messages": messages, "temperature": 0.1},
                timeout=45,
            )
            if not response.ok:
                try:
                    provider_error = response.json().get("error", {}).get("message", response.text)
                except ValueError:
                    provider_error = response.text
                raise RuntimeError(f"Groq returned HTTP {response.status_code}: {provider_error}")
            body = response.json()
            return body["choices"][0]["message"]["content"], "Groq"

        if config.get("gemini_api_key"):
            contents = [{"role": "user", "parts": [{"text": f"{system}\n\n{context}\n\nQuestion: {question}"}]}]
            configured_model = os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.8-flash"
            models = [configured_model]
            for fallback_model in ("gemini-3.8-flash", "gemini-flash-latest", "gemini-3.1-flash-lite"):
                if fallback_model not in models:
                    models.append(fallback_model)
            last_error = "Gemini did not return an answer."
            for model in models:
                for attempt in range(2):
                    try:
                        response = requests.post(
                            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                            params={"key": config["gemini_api_key"].strip()},
                            json={"contents": contents, "generationConfig": {"temperature": 0.1}},
                            timeout=20,
                        )
                    except requests.RequestException as exc:
                        last_error = f"Gemini request failed for {model}: {exc}"
                        if attempt == 1:
                            break
                        import time

                        time.sleep(1.5)
                        continue
                    if response.status_code != 503 or attempt == 1:
                        break
                    import time

                    time.sleep(1.5)
                if response.ok:
                    try:
                        return response.json()["candidates"][0]["content"]["parts"][0]["text"], f"Gemini ({model})"
                    except (KeyError, IndexError, TypeError) as exc:
                        raise RuntimeError("Gemini returned an unexpected response shape.") from exc
                try:
                    provider_error = response.json().get("error", {}).get("message", response.text)
                except ValueError:
                    provider_error = response.text
                last_error = f"Gemini returned HTTP {response.status_code}: {provider_error}"
                if response.status_code not in {404, 503}:
                    break
            raise RuntimeError(last_error)

        return self._local_answer(question, citations_from_context(context)), "Local retrieval"

    def _local_answer(self, question: str, citations: list[dict[str, Any]]) -> str:
        terms = {term.lower() for term in re.findall(r"\w+", question) if len(term) > 2}
        ranked = sorted(
            citations,
            key=lambda item: sum(term in item["text"].lower() for term in terms),
            reverse=True,
        )
        excerpts = [item["text"].strip().replace("\n", " ") for item in ranked[:2] if item["text"].strip()]
        if not excerpts:
            return "I could not find a relevant answer in the indexed documents."
        return "I found these relevant passages in your documents:\n\n" + "\n\n".join(f"- {excerpt}" for excerpt in excerpts)

    def ask(self, question: str, history: list[dict[str, str]], config: dict[str, str | None]) -> dict[str, Any]:
        citations = self._retrieve(question, config)
        provider_error = None
        try:
            answer, provider = self._hosted_answer(question, self._context(citations), history, config)
        except (RuntimeError, requests.RequestException) as exc:
            provider_error = str(exc)
            answer = self._local_answer(question, citations)
            provider = "Local retrieval (hosted provider unavailable)"
        return {"answer": answer, "provider": provider, "provider_error": provider_error, "citations": citations}

    def list_documents(self) -> list[dict[str, Any]]:
        documents = []
        for path in sorted(self.upload_directory.glob("*")):
            if path.suffix.lower() in {".pdf", ".txt"}:
                documents.append({"name": path.name, "size": path.stat().st_size})
        return documents

    def delete_document(self, filename: str) -> dict[str, Any]:
        safe_name = Path(filename).name
        path = self.upload_directory / safe_name
        if not path.is_file() or path.suffix.lower() not in {".pdf", ".txt"}:
            raise ValueError(f"Document not found: {safe_name}")

        if self._vectorstore is not None:
            self._vectorstore.delete(where={"source": safe_name})
        path.unlink()
        return {"name": safe_name}

    def clear(self) -> None:
        if self._vectorstore is not None:
            self._vectorstore.delete_collection()
        self._vectorstore = None
        self._embedding_signature = None
        if self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)
        for path in self.upload_directory.iterdir():
            if path.is_file():
                path.unlink()


def citations_from_context(context: str) -> list[dict[str, Any]]:
    """Build the minimal citation shape needed by the deterministic local fallback."""
    citations = []
    for block in context.split("\n\n"):
        match = re.match(r"\[Source: (.*?), page (.*?)\]\n(.*)", block, re.DOTALL)
        if match:
            citations.append({"source": match.group(1), "page": match.group(2), "text": match.group(3)})
    return citations
