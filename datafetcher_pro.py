# datafetcher_pro.py
import os
import io
import time
import json
import hashlib
import zipfile
import gzip
import tarfile
import logging
import requests
import warnings
import pandas as pd
import xml.etree.ElementTree as etree
from requests.adapters import HTTPAdapter
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Optional, Set, Dict, Any, Union, List, Tuple

__version__ = "0.1.0"
__all__ = ["DataFetcherPro"]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def warn(*args, **kwargs):
    """No-op warn for optional global suppression."""
    pass


class DataFetcherPro:
    """
    DataFetcherPro - Robust downloader + loader for data files (ETL helper).

    Features:
      - Download with retries and optional headers/auth.
      - Validate HTTP status and basic integrity (content-length or hash).
      - Support for compressed containers (.zip, .tar, .gz) with extraction.
      - Option to read into memory (no file write) or save locally.
      - Load many file formats into pandas DataFrame (csv, excel, json, xml, parquet, feather, hdf5, stata, sas, pickle).
      - CSV chunked reading, read params passthrough, dtype conversion.
      - Convenience methods: fetch(), summary(), save(), append(), merge(), __repr__.
      - Simple web scraping: fetch_html_table() and helpers.
      - API helper: fetch_api_json().
    """

    def __init__(
        self,
        url: str,
        filename: Optional[str] = None,
        *,
        suppress_warnings: bool = True,
        headers: Optional[Dict[str, str]] = None,
        auth: Optional[Union[Tuple[str, str], requests.auth.AuthBase]] = None,
        timeout: int = 30,
        request_delay: float = 0.0,
        max_retries: int = 3,
        backoff_factor: float = 0.3,
    ):
        """
        Parameters:
            url: source URL (must start with http:// or https://).
            filename: local filename to save as (if None, derived from URL).
            suppress_warnings: disable python warnings if True.
            headers: optional HTTP headers (e.g., for tokens).
            auth: auth tuple or requests Auth object.
            timeout: request timeout in seconds.
            request_delay: delay between requests in seconds (rate limiting).
            max_retries: number of retry attempts for transient failures.
            backoff_factor: backoff multiplier for retries.
        """
        # Basic URL validation early
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {url}")

        self.url = url

        # If no filename provided, derive from URL
        if filename is None:
            filename = os.path.basename(url.split("?")[0]) or "downloaded_file"
        self.filename = filename

        # filepath as Path
        self.filepath: Path = Path.cwd() / self.filename

        # Data & buffers
        self.df: Optional[pd.DataFrame] = None
        self._buffer: Optional[bytes] = None
        self.extracted_paths: List[Path] = []

        # HTTP / session options
        self.headers = headers
        self.auth = auth
        self.timeout = timeout
        self.request_delay = request_delay

        # Setup requests session with retries
        self.session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]),
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Optionally suppress warnings (local context)
        if suppress_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

    # --------------------------
    # DOWNLOAD / FETCH
    # --------------------------
    def download(
        self,
        save: bool = True,
        expected_size: Optional[int] = None,
        expected_hash: Optional[str] = None,
        hash_algo: str = "md5",
        verify_ssl: bool = True,
        stream: bool = True,
        extract: bool = False,
    ) -> Optional[Path]:
        """
        Download the resource.

        Parameters:
            save: whether to write to disk (True) or keep in memory (False).
            expected_size: optional number of bytes expected (validate content-length).
            expected_hash: optional hex hash string to validate content (md5 by default).
            hash_algo: 'md5'|'sha1'|'sha256', etc.
            verify_ssl: pass to requests.get verify.
            stream: stream request (True recommended for large files).
            extract: if True and file is archive (zip/tar/gz) try to extract.

        Returns:
            Path to saved file if save=True, else None if content kept in memory.
        """
        # If already present on disk, return early
        if save and self.filepath.exists():
            logger.info(f"[download] File already exists at: {self.filepath}")
            return self.filepath

        # Perform GET with session + retries
        try:
            resp = self.session.get(
                self.url,
                headers=self.headers,
                auth=self.auth,
                timeout=self.timeout,
                stream=stream,
                verify=verify_ssl,
            )
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"[download] HTTP request failed for {self.url}: {e}") from e

        # Optional validation of content-length
        if expected_size is not None:
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) != expected_size:
                raise RuntimeError(
                    f"[download] Expected size {expected_size} but server reported {content_length}"
                )

        # Read content
        if stream:
            chunks = []
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    chunks.append(chunk)
            content = b"".join(chunks)
        else:
            content = resp.content

        # Optional hash validation
        if expected_hash is not None:
            h = hashlib.new(hash_algo)
            h.update(content)
            digest = h.hexdigest()
            if digest != expected_hash:
                raise RuntimeError(
                    f"[download] Hash mismatch: expected {expected_hash} got {digest}"
                )

        # Save to disk or keep in memory
        if save:
            # ensure parent directory exists
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, "wb") as f:
                f.write(content)
            logger.info(f"[download] File saved at: {self.filepath}")
            if extract:
                self._maybe_extract_file(self.filepath)
            return self.filepath
        else:
            self._buffer = content
            logger.info("[download] Content stored in memory (no file written).")
            if extract:
                with io.BytesIO(content) as b:
                    self._maybe_extract_bytes(b, self.filename)
            return None

    # --------------------------
    # Extraction helpers (Path-aware)
    # --------------------------
    def _maybe_extract_file(self, path: Path):
        """
        If path is zip/tar/gz, extract contents into folder <filename>_extracted.
        Updates self.extracted_paths as list of Path objects.
        """
        base_dir = path.with_name(path.stem + "_extracted")

        # ZIP
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, "r") as z:
                base_dir.mkdir(parents=True, exist_ok=True)
                z.extractall(base_dir)
                self.extracted_paths = [base_dir / n for n in z.namelist()]
            logger.info(f"[extract] ZIP extracted to {base_dir}")

        # GZ (single-file)
        elif path.suffix == ".gz" and not zipfile.is_zipfile(path):
            out_path = path.with_suffix("")  # remove .gz
            with gzip.open(path, "rb") as gz_in:
                with open(out_path, "wb") as out_f:
                    out_f.write(gz_in.read())
            self.extracted_paths = [out_path]
            logger.info(f"[extract] GZ decompressed to {out_path}")

        # TAR
        elif tarfile.is_tarfile(path):
            with tarfile.open(path, "r:*") as t:
                base_dir.mkdir(parents=True, exist_ok=True)
                t.extractall(base_dir)
                self.extracted_paths = [base_dir / m.name for m in t.getmembers() if m.isreg()]
            logger.info(f"[extract] TAR extracted to {base_dir}")

        else:
            self.extracted_paths = []

    def _maybe_extract_bytes(self, b: io.BytesIO, tmp_name: str):
        """
        Try to extract in-memory archive bytes; writes files to <tmp_name>_extracted.
        """
        base_dir = Path(tmp_name + "_extracted")
        b.seek(0)

        # Try as ZIP
        try:
            with zipfile.ZipFile(b) as z:
                base_dir.mkdir(parents=True, exist_ok=True)
                z.extractall(base_dir)
                self.extracted_paths = [base_dir / n for n in z.namelist()]
            logger.info(f"[extract] In-memory ZIP extracted to {base_dir}")
            return
        except Exception:
            b.seek(0)

        # Try as TAR
        try:
            with tarfile.open(fileobj=b) as t:
                base_dir.mkdir(parents=True, exist_ok=True)
                t.extractall(base_dir)
                self.extracted_paths = [base_dir / m.name for m in t.getmembers() if m.isreg()]
            logger.info(f"[extract] In-memory TAR extracted to {base_dir}")
            return
        except Exception:
            b.seek(0)

        # Try as GZ (single-file)
        try:
            with gzip.open(b, "rb") as gz_in:
                base_dir.mkdir(parents=True, exist_ok=True)
                out_path = base_dir / tmp_name.rstrip(".gz")
                with open(out_path, "wb") as out_f:
                    out_f.write(gz_in.read())
                self.extracted_paths = [out_path]
            logger.info(f"[extract] In-memory GZ decompressed to {out_path}")
        except Exception:
            self.extracted_paths = []

    # --------------------------
    # WEB SCRAPING / API HELPERS
    # --------------------------
    def _respect_robots(self, target_url: Optional[str] = None, user_agent: str = "*") -> bool:
        """
        Check robots.txt for permission to fetch target_url.
        Returns True if allowed or robots.txt absent/unparseable.
        """
        target = target_url or self.url
        parsed = urlparse(target)
        robots_url = urljoin(f"{parsed.scheme}://{parsed.netloc}", "/robots.txt")
        rp = RobotFileParser()
        try:
            rp.set_url(robots_url)
            rp.read()
            return rp.can_fetch(user_agent, target)
        except Exception:
            # If robots.txt unavailable or failed to parse, be permissive
            return True

    def _safe_get(self, url: str, **kwargs) -> requests.Response:
        """
        Internal wrapper to perform GET with optional delay, headers and retries via session.
        Respects robots.txt and applies rate limiting.
        """
        if not self._respect_robots(url):
            raise RuntimeError(f"[scrape] Fetch disallowed by robots.txt: {url}")

        # rate limiting delay if configured on instance
        if self.request_delay and self.request_delay > 0:
            time.sleep(self.request_delay)

        # Merge instance headers with any passed kwargs headers
        merged_headers = dict(self.headers) if self.headers else {}
        if "headers" in kwargs:
            merged_headers.update(kwargs["headers"])
            kwargs["headers"] = merged_headers
        else:
            kwargs["headers"] = merged_headers

        resp = self.session.get(url, auth=self.auth, timeout=self.timeout, **kwargs)
        resp.raise_for_status()
        return resp

    def fetch_html_tables(
        self, url: Optional[str] = None, attrs: Optional[Dict[str, str]] = None, header: int = 0
    ) -> List[pd.DataFrame]:
        """
        Return list of tables found on page as DataFrames.
        Uses pandas.read_html first, falls back to BeautifulSoup.
        """
        target = url or self.url
        resp = self._safe_get(target)
        try:
            tables = pd.read_html(resp.text, attrs=attrs, header=header)
            return tables
        except Exception:
            # fallback parsing with BeautifulSoup
            soup = BeautifulSoup(resp.text, "lxml")
            tables = soup.find_all("table")
            dfs: List[pd.DataFrame] = []
            for tbl in tables:
                try:
                    df = pd.read_html(str(tbl), header=header)[0]
                    dfs.append(df)
                except Exception:
                    continue
            if not dfs:
                raise RuntimeError("[fetch_html_tables] No tables found or parsing failed.")
            return dfs

    def fetch_html_table_by_selector(
        self, url: Optional[str] = None, selector: str = "table", table_index: int = 0, header: int = 0
    ) -> pd.DataFrame:
        """
        Find element(s) matching CSS selector and convert the selected table to DataFrame.
        Useful for pages with multiple tables or specific class/id.
        """
        target = url or self.url
        resp = self._safe_get(target)
        soup = BeautifulSoup(resp.text, "lxml")
        elems = soup.select(selector)
        if not elems:
            raise RuntimeError(f"[fetch_html_table_by_selector] No elements found for selector: {selector}")
        try:
            html_fragment = str(elems[table_index])
            df = pd.read_html(html_fragment, header=header)[0]
            return df
        except Exception as e:
            raise RuntimeError(f"[fetch_html_table_by_selector] Failed to parse selected table: {e}") from e

    def extract_links(
        self, url: Optional[str] = None, selector: Optional[str] = None, unique: bool = True, absolute: bool = True
    ) -> List[str]:
        """
        Extract href links from page. Optionally pass a CSS selector to narrow scope.
        Returns list of URLs (absolute if absolute=True).
        """
        target = url or self.url
        resp = self._safe_get(target)
        soup = BeautifulSoup(resp.text, "lxml")
        scope = soup.select(selector) if selector else [soup]
        links: List[str] = []
        for node in scope:
            for a in node.find_all("a", href=True):
                href = a["href"].strip()
                if absolute:
                    href = urljoin(target, href)
                links.append(href)
        if unique:
            seen: Set[str] = set()
            ordered: List[str] = []
            for u in links:
                if u not in seen:
                    seen.add(u)
                    ordered.append(u)
            return ordered
        return links

    def extract_metadata(self, url: Optional[str] = None) -> Dict[str, Optional[str]]:
        """
        Extract basic page metadata: title, description, canonical, og:image, keywords.
        """
        target = url or self.url
        resp = self._safe_get(target)
        soup = BeautifulSoup(resp.text, "lxml")

        def get_meta(name=None, prop=None):
            if name:
                tag = soup.find("meta", attrs={"name": name})
            else:
                tag = soup.find("meta", attrs={"property": prop})
            return tag["content"].strip() if tag and tag.get("content") else None

        canonical_link = soup.find("link", rel="canonical")
        meta = {
            "title": soup.title.string.strip() if soup.title and soup.title.string else None,
            "description": get_meta(name="description"),
            "canonical": canonical_link.get("href") if canonical_link else None,
            "og:image": get_meta(prop="og:image"),
            "keywords": get_meta(name="keywords"),
        }
        return meta

    def fetch_text(self, url: Optional[str] = None, selector: Optional[str] = None, collapse_whitespace: bool = True) -> str:
        """
        Return cleaned visible text from the page or a sub-element (selected by CSS).
        Removes scripts/styles and collapses whitespace.
        """
        target = url or self.url
        resp = self._safe_get(target)
        soup = BeautifulSoup(resp.text, "lxml")
        if selector:
            el = soup.select_one(selector)
            if not el:
                raise RuntimeError(f"[fetch_text] No element matches selector: {selector}")
            text = el.get_text(separator=" ", strip=True)
        else:
            for s in soup(["script", "style", "noscript"]):
                s.decompose()
            text = soup.get_text(separator=" ", strip=True)
        if collapse_whitespace:
            text = " ".join(text.split())
        return text

    def render_js_page(self, url: Optional[str] = None, wait_for: float = 2.0, selector: Optional[str] = None) -> str:
        """
        Render a JS-heavy page and return HTML. Requires Playwright installed.
        """
        target = url or self.url
        try:
            from playwright.sync_api import sync_playwright
        except Exception as e:
            raise RuntimeError(
                "[render_js_page] Playwright not installed. Install with `pip install playwright` and run `playwright install`"
            ) from e

        if not self._respect_robots(target):
            raise RuntimeError(f"[render_js_page] Fetch disallowed by robots.txt: {target}")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(target, timeout=int(self.timeout * 1000))
            if selector:
                try:
                    page.wait_for_selector(selector, timeout=int(wait_for * 1000))
                except Exception:
                    pass
            else:
                page.wait_for_timeout(int(wait_for * 1000))
            html = page.content()
            browser.close()
            return html

    # --------------------------
    # LOADING
    # --------------------------
    def load(
        self,
        *,
        from_buffer: bool = False,
        chunksize: Optional[int] = None,
        delimiter: Optional[str] = None,
        encoding: Optional[str] = None,
        sheet_name: Optional[Union[str, int]] = 0,
        dtype: Optional[Dict[str, Any]] = None,
        parse_dates: Optional[Union[List[str], Dict[str, Any], bool]] = False,
        low_memory: bool = True,
        **read_kwargs,
    ) -> pd.DataFrame:
        """
        Load the file (saved on disk or from in-memory buffer) into a DataFrame.
        Pass optional read parameters (delimiter, encoding, sheet_name, dtype, parse_dates, chunksize).

        If chunksize is provided for CSV, returns concatenated DataFrame by default.

        read_kwargs: passthrough to pandas readers.
        """
        # Determine source: buffer or file
        if from_buffer:
            if self._buffer is None:
                raise RuntimeError("[load] No in-memory buffer found. Use download(save=False) first.")
            content_source = io.BytesIO(self._buffer)
            source_is_buffer = True
        else:
            if not self.filepath.exists():
                raise RuntimeError(f"[load] File not found at {self.filepath}. Run download() first or set save=True.")
            content_source = self.filepath
            source_is_buffer = False

        # Determine extension from filename or filepath
        ext = self.filepath.suffix.lower() or Path(self.filename).suffix.lower()

        try:
            if ext == ".csv" or (ext == "" and source_is_buffer):
                # If reading from buffer, use the BytesIO directly
                if source_is_buffer:
                    content_source.seek(0)
                    df_iter = pd.read_csv(
                        content_source,
                        delimiter=delimiter,
                        encoding=encoding,
                        dtype=dtype,
                        parse_dates=parse_dates,
                        low_memory=low_memory,
                        chunksize=chunksize,
                        **read_kwargs,
                    )
                else:
                    df_iter = pd.read_csv(
                        str(self.filepath),
                        delimiter=delimiter,
                        encoding=encoding,
                        dtype=dtype,
                        parse_dates=parse_dates,
                        low_memory=low_memory,
                        chunksize=chunksize,
                        **read_kwargs,
                    )

                if chunksize:
                    df = pd.concat(df_iter, ignore_index=True)
                else:
                    df = df_iter
                self.df = df

            elif ext in [".xlsx", ".xls"]:
                self.df = pd.read_excel(content_source, sheet_name=sheet_name, dtype=dtype, **read_kwargs)

            elif ext == ".json":
                if source_is_buffer:
                    content_source.seek(0)
                    text = content_source.read().decode(encoding or "utf-8")
                    data = json.loads(text)
                else:
                    with open(self.filepath, "r", encoding=encoding or "utf-8") as f:
                        data = json.load(f)
                if isinstance(data, dict):
                    self.df = pd.json_normalize(data)
                else:
                    self.df = pd.DataFrame(data)

            elif ext == ".xml":
                try:
                    if source_is_buffer:
                        content_source.seek(0)
                        self.df = pd.read_xml(content_source)
                    else:
                        self.df = pd.read_xml(str(self.filepath))
                except Exception:
                    # fallback manual xml parse
                    if source_is_buffer:
                        content_source.seek(0)
                        tree = etree.parse(content_source)
                    else:
                        tree = etree.parse(str(self.filepath))
                    root = tree.getroot()
                    rows: List[List[Any]] = []
                    cols = [child.tag for child in root[0]]
                    for node in root:
                        rows.append([child.text for child in node])
                    self.df = pd.DataFrame(rows, columns=cols)

            elif ext == ".parquet":
                self.df = pd.read_parquet(content_source)

            elif ext == ".feather":
                self.df = pd.read_feather(content_source)

            elif ext in [".h5", ".hdf5"]:
                self.df = pd.read_hdf(content_source, **read_kwargs)

            elif ext == ".dta":
                self.df = pd.read_stata(content_source)

            elif ext == ".sas7bdat":
                self.df = pd.read_sas(content_source, format="sas7bdat")

            elif ext == ".pkl":
                self.df = pd.read_pickle(content_source)

            else:
                raise ValueError(f"[load] File type '{ext}' not supported")

            logger.info(f"[load] Data loaded into DataFrame with shape: {self.df.shape}")
            return self.df

        except Exception as e:
            raise RuntimeError(f"[load] Failed to load file {self.filename}: {e}") from e

    # --------------------------
    # CONVENIENCE / AUX METHODS
    # --------------------------
    def fetch(self, save: bool = True, **download_kwargs) -> pd.DataFrame:
        """
        High-level convenience: download (save or not) and load.

        Parameters:
          save: whether to save to disk (True) or keep in memory (False).
          download_kwargs: forwarded to download()
        """
        self.download(save=save, **download_kwargs)
        return self.load(from_buffer=not save)

    def summary(self, n_head: int = 5):
        """Print simple DataFrame summary (info, head, describe)."""
        if self.df is None:
            logger.warning("[summary] No DataFrame loaded. Use fetch() or load() first.")
            return
        print("=== INFO ===")
        self.df.info()
        print("\n=== HEAD ===")
        print(self.df.head(n_head))
        print("\n=== DESCRIBE ===")
        print(self.df.describe(include="all"))

    def save(self, out_path: Union[str, Path], fmt: Optional[str] = None, **save_kwargs):
        """
        Save the current DataFrame to disk in a preferred format.
        """
        if self.df is None:
            raise RuntimeError("[save] No DataFrame loaded to save.")
        out_path = Path(out_path)
        if fmt is None:
            ext = out_path.suffix.lower()
            fmt = ext.lstrip(".")
        if fmt == "csv":
            self.df.to_csv(out_path, index=False, **save_kwargs)
        elif fmt in ("xlsx", "xls", "excel"):
            self.df.to_excel(out_path, index=False, **save_kwargs)
        elif fmt == "parquet":
            self.df.to_parquet(out_path, index=False, **save_kwargs)
        elif fmt in ("pkl", "pickle"):
            self.df.to_pickle(out_path, **save_kwargs)
        else:
            try:
                writer = getattr(self.df, f"to_{fmt}")
                writer(out_path, **save_kwargs)
            except Exception as e:
                raise RuntimeError(f"[save] Unsupported format or save failed: {e}") from e
        logger.info(f"[save] DataFrame saved to {out_path}")

    def append(self, other: Union["DataFetcherPro", pd.DataFrame]):
        """
        Append another DataFrame or DataFetcherPro's DataFrame to this one (in-place).
        """
        if isinstance(other, DataFetcherPro):
            other_df = other.df
        elif isinstance(other, pd.DataFrame):
            other_df = other
        else:
            raise ValueError("[append] other must be DataFetcherPro or pandas.DataFrame")
        if other_df is None:
            raise RuntimeError("[append] Other has no DataFrame loaded.")
        if self.df is None:
            self.df = other_df.copy()
        else:
            self.df = pd.concat([self.df, other_df], ignore_index=True)
        logger.info("[append] DataFrames concatenated.")

    def merge(self, other: Union["DataFetcherPro", pd.DataFrame], **merge_kwargs) -> pd.DataFrame:
        """
        Merge this DataFrame with another on specified keys. Returns new DataFrame.
        """
        if isinstance(other, DataFetcherPro):
            other_df = other.df
        elif isinstance(other, pd.DataFrame):
            other_df = other
        else:
            raise ValueError("[merge] other must be DataFetcherPro or pandas.DataFrame")
        if self.df is None or other_df is None:
            raise RuntimeError("[merge] Both DataFrames must be loaded.")
        result = pd.merge(self.df, other_df, **merge_kwargs)
        return result

    # --------------------------
    # WEB SCRAPING / API HELPERS
    # --------------------------
    def fetch_html_table(self, url: Optional[str] = None, table_index: int = 0, attrs: Optional[Dict[str, str]] = None, header: int = 0) -> pd.DataFrame:
        """
        Fetch an HTML page and parse a table into DataFrame.
        """
        target = url or self.url
        resp = self._safe_get(target)
        try:
            tables = pd.read_html(resp.text, attrs=attrs, header=header)
            df = tables[table_index]
            return df
        except Exception:
            try:
                soup = BeautifulSoup(resp.text, "lxml")
                tables = soup.find_all("table")
                if not tables:
                    raise RuntimeError("[fetch_html_table] No <table> found on page.")
                tbl = tables[table_index]
                df = pd.read_html(str(tbl), header=header)[0]
                return df
            except Exception as e2:
                raise RuntimeError(f"[fetch_html_table] Failed to parse HTML table: {e2}") from e2

    def fetch_api_json(self, params: Optional[Dict[str, Any]] = None, method: str = "GET", json_body: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Call a JSON API and parse the returned JSON into a DataFrame.
        """
        try:
            if method.upper() == "GET":
                resp = self._safe_get(self.url, params=params)
            else:
                if self.request_delay and self.request_delay > 0:
                    time.sleep(self.request_delay)
                resp = self.session.request(method.upper(), self.url, headers=self.headers, auth=self.auth, json=json_body, params=params, timeout=self.timeout)
                resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                df = pd.json_normalize(data)
            else:
                df = pd.DataFrame(data)
            self.df = df
            return df
        except Exception as e:
            raise RuntimeError(f"[fetch_api_json] API request failed: {e}") from e

    # --------------------------
    # REPRESENTATION
    # --------------------------
    def __repr__(self) -> str:
        return f"<DataFetcherPro url={self.url!r} filename={self.filename!r} loaded={self.df is not None}>"

    def __str__(self) -> str:
        return f"DataFetcherPro for {self.filename} ({'loaded' if self.df is not None else 'not loaded'})"


