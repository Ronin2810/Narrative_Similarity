import os
import re
import pandas as pd
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    raise ImportError("pip install pdfplumber")

DATA_ROOT = Path("data")
OUTPUT_CSV = "transcripts_clean.csv"


def extract_text_from_pdf(pdf_path: Path) -> str:
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text_chunks.append(t)
    full_text = "\n".join(text_chunks)
    # basic cleanup
    full_text = full_text.replace("\r", " ").strip()
    return full_text


def parse_year_quarter_from_name(name: str):
    base = name.lower()

    m = re.search(r"(q[1-4])[^0-9]{0,3}(20[0-9]{2})", base)
    if m:
        q = m.group(1).upper()          
        year = int(m.group(2))          
        return year, q

    m2 = re.search(r"(20[0-9]{2})[^0-9]{0,6}earnings", base)
    if m2:
        year = int(m2.group(1))
        return year, "Q4"

    m3 = re.findall(r"(20[0-9]{2})", base)
    if m3:
        year = int(m3[-1])
        return year, None

    return None, None


def main():
    rows = []

    for ticker_dir in DATA_ROOT.iterdir():
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name.upper()

        for pdf_path in ticker_dir.glob("*.pdf"):
            text = extract_text_from_pdf(pdf_path)

            year, quarter = parse_year_quarter_from_name(pdf_path.name)
            if year is None:
                print(f"Could not parse year/quarter from {pdf_path.name}")
                continue

            period = f"{year}{quarter}" if quarter is not None else str(year)

            rows.append({
                "ticker": ticker,
                "year": year,
                "quarter": quarter,
                "period": period,
                "source_file": pdf_path.name,
                "clean_text": text
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    main()