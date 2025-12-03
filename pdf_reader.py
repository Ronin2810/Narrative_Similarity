from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List


from collections import Counter
from pypdf import PdfReader
import re

@dataclass
class Document:
    ticker: str
    doc_id: str      # e.g. filename
    text: str        # full extracted text
    period: str = ""  # optional period attribute


def split_pages(text):
    # If your file uses '\f' page breaks from PDF → TXT, this works.
    # Otherwise you might need another rule (e.g. "Page X of Y" lines).
    return text.split("\f")

def find_common_header_footer(pages, header_lines=5, footer_lines=5, threshold=0.8):
    header_counter = Counter()
    footer_counter = Counter()

    for page in pages:
        lines = page.splitlines()
        if not lines:
            continue

        header_part = lines[:header_lines]
        footer_part = lines[-footer_lines:]

        header_counter.update([l.strip() for l in header_part if l.strip()])
        footer_counter.update([l.strip() for l in footer_part if l.strip()])

    n_pages = max(len(pages), 1)

    common_header = {
        line for line, cnt in header_counter.items()
        if cnt / n_pages >= threshold
    }
    common_footer = {
        line for line, cnt in footer_counter.items()
        if cnt / n_pages >= threshold
    }

    return common_header, common_footer

def remove_header_footer(text, header_lines=5, footer_lines=5, threshold=0.8):
    pages = split_pages(text)
    common_header, common_footer = find_common_header_footer(
        pages, header_lines, footer_lines, threshold
    )

    DISCLAIMER_START_PATTERNS = [
    r"^DISCLAIMER\s*$",
    r"^Thomson Reuters reserves the right to make changes to documents",
    # r"^\(1-877-322-7338\)",
    # r"^www\.factset\.com",

    ]


    cleaned_pages = []
    for page in pages:
        lines = page.splitlines()
        new_lines = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Check for disclaimer start
            if any(re.match(pattern, stripped, re.IGNORECASE) for pattern in DISCLAIMER_START_PATTERNS):
                break  # Stop processing further lines in this page
            # If it matches a known header/footer line, skip it
            if stripped in common_header or stripped in common_footer:
                continue
                
            if 'factset' in stripped.lower():
                continue

            new_lines.append(line)
        cleaned_pages.append("\n".join(new_lines))
    


    return "\n\f\n".join(cleaned_pages)


def extract_text_from_pdf(path: Path) -> str:
    """Extract plain text from a PDF file using pypdf."""
    reader = PdfReader(str(path))
    pages_text: List[str] = []
    period = re.search(r'(Q[1-4] \d{4})', reader.pages[0].extract_text()).group(0) if re.search(r'(Q[1-4] \d{4})', reader.pages[0].extract_text()) else ""
    for page in reader.pages:
        # .extract_text() can return None sometimes; guard it
        page_text = page.extract_text() or ""
        page_text = remove_header_footer(page_text)
        pages_text.append(page_text)
    return "\n\f".join(pages_text), period


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
            text, period = extract_text_from_pdf(pdf_path)
            if not text.strip():
                # Optionally skip empty PDFs
                continue

            yield Document(
                ticker=ticker,
                doc_id=pdf_path.name,
                text=text,
                #Period is first instance of Q# #### in text
                period = period
            )
