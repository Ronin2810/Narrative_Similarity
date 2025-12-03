import requests

from bs4 import BeautifulSoup
import re
import os
from urllib.parse import urlparse, unquote

def get_coca_cola_links():
    url = 'https://investors.coca-colacompany.com/financial-information'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    transcripts = soup.find_all('a', class_='result-link')
    # Make sure we return absolute URLs
    from urllib.parse import urljoin
    transcript_links = [urljoin(url, transcript.get('href', '')) for transcript in transcripts]
    transcript_links = list(filter(lambda x: 'transcript' in x, transcript_links))
    return transcript_links

def get_pepsi_links():
    url = 'https://www.pepsico.com/investors/earnings'
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    response = session.get(url, headers=headers, stream=True, allow_redirects=True, timeout=30)
    print(session.cookies)
    # response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    print(response.content)
    divs = soup.find_all('div', class_='rte')
    print(divs)
    links = []
    for div in divs:
        a_tags = div.find_all('a')
        for a in a_tags:
            href = a.get('href', '')
            links.append(href)
    print(links)
    # Make sure we return absolute URLs
    from urllib.parse import urljoin
    transcript_links = [urljoin(url,transcript) for transcript in links]
    transcript_links = list(filter(lambda x: 'transcript' in x, transcript_links))
    return transcript_links

def get_walmart_links():
    url = 'https://stock.walmart.com/investors/financial-information/earnings/default.aspx'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    transcripts = soup.find_all('div', class_='result-line')
    transcript_links = [urljoin(url, transcript.find('a').get('href', '')) for transcript in transcripts]
    transcript_links = list(filter(lambda x: 'transcript' in x, transcript_links))
    # Make sure we return absolute URLs
    from urllib.parse import urljoin


def get_sanitized_filename(url, response, parent_folder):
    
    # Try to extract filename from Content-Disposition header
    fname = None
    cd = response.headers.get('content-disposition')
    if cd:
        # RFC6266 allows either filename*=UTF-8''urlencoded or filename="..."
        m = re.search(r"filename\*=(?:UTF-8'')?([^;]+)", cd, flags=re.IGNORECASE)
        if m:
            fname = m.group(1).strip().strip('"')
            # If url-encoded (from filename*), unquote it
            fname = unquote(fname)
        else:
            m2 = re.search(r'filename="?([^";]+)"?', cd)
            if m2:
                fname = m2.group(1).strip()

    # Fallback: derive filename from URL path
    if not fname:
        path = urlparse(url).path
        fname = os.path.basename(unquote(path)) or 'downloaded_file'

    # Sanitize filename for Windows/posix (remove/replace problematic chars)
    invalid_chars = '<>:\\"/|?* ,'
    for ch in invalid_chars:
        fname = fname.replace(ch, '_')
    fname = fname.strip()

    out_path = os.path.join(parent_folder, fname)
    return out_path

def download_file(url, parent_folder):
    # Ensure parent folder exists
    os.makedirs(parent_folder, exist_ok=True)
    # Use a session and stream the download to avoid loading large bodies into memory
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    # Some sites behave differently without a referer; set one using the origin of the URL
    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc:
        headers['Referer'] = f"{parsed.scheme}://{parsed.netloc}/"

    resp = session.get(url, headers=headers, stream=True, allow_redirects=True, timeout=30)
    try:
        resp.raise_for_status()
    except Exception:
        # Save a small debug file with headers and first bytes to help diagnosis
        out_path = get_sanitized_filename(url, resp, parent_folder)
        dbg_path = out_path + '.debug.html'
        with open(dbg_path, 'wb') as dbg:
            dbg.write(b"<!-- HTTP ERROR -->\n")
            dbg.write(str(resp.status_code).encode('utf-8') + b"\n")
            dbg.write(str(resp.headers).encode('utf-8') + b"\n\n")
            dbg.write(resp.content[:4096])
        raise

    out_path = get_sanitized_filename(url, resp, parent_folder)

    content_type = resp.headers.get('content-type', '')
    content_length = resp.headers.get('content-length')
    print(f"Downloading -> {out_path}  (Content-Type: {content_type}; Content-Length: {content_length})")

    # Stream-write into file
    written = 0
    with open(out_path, 'wb') as file:
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                continue
            file.write(chunk)
            written += len(chunk)

    # Basic sanity checks
    if written == 0:
        # Save first bytes for debugging
        dbg_path = out_path + '.debug.html'
        with open(dbg_path, 'wb') as dbg:
            dbg.write(resp.content[:4096])
        print(f"Warning: wrote 0 bytes to {out_path}. Debug saved to {dbg_path}")
    else:
        print(f"Wrote {written} bytes to {out_path}")
    
def download_coca_cola_transcripts():
    links = get_coca_cola_links()
    # print(links)
    for link in links:
        print(f'Downloading {link}...')
        download_file(link, parent_folder='data\\KO') # Run from ./CS573

def download_pepsi_transcripts():
    links = get_pepsi_links()
    # print(links)
    for link in links:
        print(f'Downloading {link}...')
        download_file(link, parent_folder='data\\PEP') # Run from ./CS573

if __name__ == '__main__':
    # download_coca_cola_transcripts()
    download_pepsi_transcripts()
    