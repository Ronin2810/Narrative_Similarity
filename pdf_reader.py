from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

from pypdf import PdfReader

@dataclass
class Document:
    ticker: str
    doc_id: str      # e.g. filename
    text: str        # full extracted text


def extract_text_from_pdf(path: Path) -> str:
    """Extract plain text from a PDF file using pypdf."""
    reader = PdfReader(str(path))
    pages_text: List[str] = []
    for page in reader.pages:
        # .extract_text() can return None sometimes; guard it
        page_text = page.extract_text() or ""
        pages_text.append(page_text)
    return "\n".join(pages_text)


def iter_documents(root_dir: Path) -> Iterator[Document]:
    """
    Walk data/{TICKER}/ and yield Document objects
    for every PDF file found.
    """
    for ticker_dir in sorted(root_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue

        ticker = ticker_dir.name.upper()

        for pdf_path in sorted(ticker_dir.glob("*.pdf")):
            text = extract_text_from_pdf(pdf_path)
            if not text.strip():
                # Optionally skip empty PDFs
                continue

            yield Document(
                ticker=ticker,
                doc_id=pdf_path.name,
                text=text,
            )
