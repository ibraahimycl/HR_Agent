import os
import re
import shutil
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY environment variable.")

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOC_PATH = os.path.join(base_dir, 'app', 'policies', 'hr_policy.txt')
CHROMA_PATH = os.path.join(base_dir, 'chroma_db')
COLLECTION_NAME = "hr_policies"


# ---------------------------------------------------------------------------
# Section-aware chunking
# ---------------------------------------------------------------------------

def split_by_policy_sections(content: str) -> List[Dict[str, str]]:
    """
    Split HR policy document respecting section boundaries.

    Strategy:
    - Each lettered subsection (A. Vacation Leave, B. Sick Leave …) → own chunk
    - Each Roman numeral section (IV. PUBLIC HOLIDAYS, VI. LEAVE ENCASHMENT …) → own chunk
    - HR POLICY MANUAL headers → own chunk
    - If a chunk exceeds 1400 chars it is split further on blank lines only,
      keeping the parent section_title.
    """
    results: List[Dict[str, str]] = []

    attendance_pos = content.find("HR POLICY MANUAL - ATTENDANCE POLICY")
    current_policy_type = "leave_policy"
    current_title = "HR Policy"
    current_lines: List[str] = []

    LETTER_SECTION = re.compile(r'^[A-Z]\.\s+[A-Za-z]')
    ROMAN_SECTION = re.compile(
        r'^(?:IV|IX|VI{0,3}|XI{0,3}|I{1,3}|V)\.\s+[A-Z]'
    )
    MANUAL_HEADER = re.compile(r'^HR POLICY MANUAL')

    def flush(title: str, lines: List[str], ptype: str):
        text = '\n'.join(lines).strip()
        if len(text) < 60:
            return
        # If section is very long, split on blank lines but keep title
        if len(text) > 1400:
            paragraphs = re.split(r'\n{2,}', text)
            for para in paragraphs:
                para = para.strip()
                if len(para) >= 60:
                    results.append({
                        "text": para,
                        "section_title": title,
                        "policy_type": ptype,
                    })
        else:
            results.append({
                "text": text,
                "section_title": title,
                "policy_type": ptype,
            })

    for raw_line in content.splitlines():
        line = raw_line.strip()

        # Update policy type when we hit attendance section
        if attendance_pos != -1 and MANUAL_HEADER.match(line) and "ATTENDANCE" in line:
            flush(current_title, current_lines, current_policy_type)
            current_policy_type = "attendance_policy"
            current_title = line[:80]
            current_lines = [line]
            continue

        is_new_section = (
            LETTER_SECTION.match(line)
            or ROMAN_SECTION.match(line)
            or MANUAL_HEADER.match(line)
        )

        if is_new_section and current_lines:
            flush(current_title, current_lines, current_policy_type)
            current_title = line[:80]
            current_lines = [line]
        else:
            current_lines.append(raw_line)

    flush(current_title, current_lines, current_policy_type)

    logger.info(f"Section-aware split produced {len(results)} chunks")
    return results


# ---------------------------------------------------------------------------
# ChromaDB creation / loading
# ---------------------------------------------------------------------------

def create_embeddings(force_rebuild: bool = False):
    """Create or load ChromaDB vector store with section-aware chunks."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

    if not force_rebuild and os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        try:
            vectorstore = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embeddings,
                collection_name=COLLECTION_NAME
            )
            count = vectorstore._collection.count()
            if count > 0:
                logger.info(
                    f"Loaded existing ChromaDB '{COLLECTION_NAME}' ({count} chunks)."
                )
                return vectorstore
        except Exception as e:
            logger.warning(f"Could not load existing ChromaDB: {e}. Re-creating...")

    # Remove stale index if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        logger.info("Removed stale chroma_db directory.")

    logger.info("Reading HR policy document...")
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        full_content = f.read()

    sections = split_by_policy_sections(full_content)

    # Inject section context directly into the text so the embedding model
    # sees it — metadata dict is not read during vector creation.
    texts = [
        f"[Policy: {s['policy_type']} | Section: {s['section_title']}]\n{s['text']}"
        for s in sections
    ]
    metadatas = [
        {
            "chunk_id": f"chunk_{i}",
            "section_title": s["section_title"],
            "policy_type": s["policy_type"],
            "doc_name": "hr_policy.txt",
        }
        for i, s in enumerate(sections)
    ]

    # Log section overview for transparency
    for m in metadatas:
        logger.info(f"  [{m['policy_type']}] {m['section_title']}")

    logger.info(f"Creating embeddings for {len(texts)} sections...")
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )
    logger.info(f"ChromaDB '{COLLECTION_NAME}' created with {len(texts)} sections.")
    return vectorstore


if __name__ == "__main__":
    create_embeddings(force_rebuild=True)
