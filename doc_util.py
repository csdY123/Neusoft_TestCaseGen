"""Document upload and management utilities"""
import os
import json
import uuid
import shutil

try:
    from docx import Document
except ImportError:
    Document = None

# Storage configuration
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploaded_docs")
DOCUMENT_INDEX_FILE = os.path.join(UPLOAD_DIR, "index.json")

# In-memory document storage
uploaded_documents = {}


def ensure_upload_dir():
    """Ensure upload directory exists"""
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def load_uploaded_documents():
    """Load uploaded documents from persistent storage"""
    ensure_upload_dir()
    uploaded_documents.clear()
    if not os.path.exists(DOCUMENT_INDEX_FILE):
        return
    try:
        with open(DOCUMENT_INDEX_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    for doc_id, doc in data.items():
        stored_name = doc.get("stored_name")
        if not stored_name:
            continue
        stored_path = os.path.join(UPLOAD_DIR, stored_name)
        display_name = doc.get("display_name") or f"{doc.get('name', 'Unnamed')} ({doc_id[:8]})"
        uploaded_documents[doc_id] = {
            "name": doc.get("name", stored_name),
            "stored_name": stored_name,
            "content": doc.get("content", ""),
            "display_name": display_name,
            "path": stored_path
        }


def persist_uploaded_documents():
    """Save uploaded documents to persistent storage"""
    ensure_upload_dir()
    serializable = {}
    for doc_id, doc in uploaded_documents.items():
        serializable[doc_id] = {
            "name": doc["name"],
            "stored_name": doc["stored_name"],
            "content": doc["content"],
            "display_name": doc["display_name"]
        }
    with open(DOCUMENT_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def extract_text_from_docx(file_path):
    """Extract text content from a .docx file"""
    if Document is None:
        raise ImportError("python-docx not installed. Please install it first.")
    try:
        document = Document(file_path)
    except Exception as exc:
        raise ValueError(f"Failed to read Word document: {exc}") from exc
    paragraphs = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def save_uploaded_document(file_data):
    """Save an uploaded document and return its ID"""
    ensure_upload_dir()
    original_name = getattr(file_data, "orig_name", None) or os.path.basename(file_data.name)
    extension = os.path.splitext(original_name)[1].lower()
    if extension != ".docx":
        raise ValueError("Only .docx format is supported.")
    
    doc_id = uuid.uuid4().hex
    stored_name = f"{doc_id}_{original_name}"
    target_path = os.path.join(UPLOAD_DIR, stored_name)
    
    try:
        shutil.copy(file_data.name, target_path)
    except OSError as exc:
        raise ValueError(f"Failed to save file: {exc}") from exc
    
    content = extract_text_from_docx(target_path)
    short_id = doc_id[:8]
    display_name = f"{original_name} ({short_id})"
    
    uploaded_documents[doc_id] = {
        "name": original_name,
        "stored_name": stored_name,
        "content": content,
        "display_name": display_name,
        "path": target_path
    }
    persist_uploaded_documents()
    return doc_id


def get_document_choices():
    """Get list of document display names for dropdown"""
    return [doc["display_name"] for doc in uploaded_documents.values()]


def get_document_by_display_name(display_name):
    """Get document ID and info by display name"""
    for doc_id, doc in uploaded_documents.items():
        if doc["display_name"] == display_name:
            return doc_id, doc
    return None, None


def get_document_content(display_name):
    """Get document content by display name"""
    _, doc = get_document_by_display_name(display_name)
    return doc["content"] if doc else ""


# Initialize on module load
load_uploaded_documents()

