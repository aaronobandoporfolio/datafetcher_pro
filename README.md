# DataFetcherPro

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DataFetcherPro** is a robust, batteries-included Python library for downloading, extracting, and loading data files into pandas DataFrames. Perfect for ETL pipelines, data analysis workflows, and web scraping tasks.

## Features

- **Smart Downloads**: Automatic retries, SSL verification, custom headers/auth, hash validation
- **Archive Support**: Auto-extract ZIP, TAR, and GZ files
- **Multiple Formats**: CSV, Excel, JSON, XML, Parquet, Feather, HDF5, Stata, SAS, Pickle
- **Memory or Disk**: Choose to save files locally or work entirely in memory
- **Web Scraping**: Built-in HTML table parsing, link extraction, metadata scraping
- **API Helper**: Simplified JSON API consumption
- **Rate Limiting**: Respect robots.txt and configure request delays
- **Chunked Reading**: Handle large CSV files efficiently

## Installation

### Basic Installation
```bash
pip install pandas requests beautifulsoup4 lxml openpyxl
```

### Full Installation (with optional dependencies)
```bash
pip install pandas requests beautifulsoup4 lxml openpyxl \
    pyarrow fastparquet tables xlrd playwright
```

### From Source
```bash
git clone https://github.com/aaronobandoporfolio/datafetcher_pro.git
cd datafetcher-pro
pip install -r requirements.txt
```

## Quick Start

### Basic Usage: Download and Load a CSV

```python
from datafetcher_pro import DataFetcherPro

# Download and load in one step
fetcher = DataFetcherPro("https://example.com/data.csv")
df = fetcher.fetch()

# View summary
fetcher.summary()
```

### Working with Excel Files

```python
fetcher = DataFetcherPro(
    "https://example.com/sales_data.xlsx",
    filename="sales.xlsx"
)
df = fetcher.fetch(save=True)

# Access specific sheet
df = fetcher.load(sheet_name="Q1_2024")
```

### Memory-Only Mode (No Disk Write)

```python
fetcher = DataFetcherPro("https://api.example.com/data.json")
df = fetcher.fetch(save=False)  # Keeps data in memory only
```

### Handling Archives

```python
# Auto-extract ZIP files
fetcher = DataFetcherPro("https://example.com/dataset.zip")
fetcher.download(save=True, extract=True)

# Access extracted files
print(fetcher.extracted_paths)  # List of Path objects
```

## Advanced Usage

### Authentication and Headers

```python
# API token authentication
fetcher = DataFetcherPro(
    "https://api.example.com/protected/data.json",
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)

# Basic auth
fetcher = DataFetcherPro(
    "https://example.com/data.csv",
    auth=("username", "password")
)
```

### Hash Validation and Retries

```python
fetcher = DataFetcherPro(
    "https://example.com/important_data.csv",
    max_retries=5,
    backoff_factor=1.0
)

# Verify file integrity
df = fetcher.fetch(
    expected_hash="5d41402abc4b2a76b9719d911017c592",
    hash_algo="md5"
)
```

### CSV Chunked Reading

```python
# Process large CSV files in chunks
fetcher = DataFetcherPro("https://example.com/huge_dataset.csv")
fetcher.download()

# Load with chunking (automatically concatenates)
df = fetcher.load(chunksize=10000)

# Or specify dtypes for efficiency
df = fetcher.load(
    dtype={"id": int, "value": float},
    parse_dates=["timestamp"]
)
```

### Web Scraping

#### Extract HTML Tables

```python
fetcher = DataFetcherPro("https://example.com/table-page.html")

# Get all tables on page
tables = fetcher.fetch_html_tables()

# Get specific table by index
df = fetcher.fetch_html_table(table_index=0)

# Select table by CSS selector
df = fetcher.fetch_html_table_by_selector(
    selector="table.data-table",
    table_index=0
)
```

#### Extract Links and Metadata

```python
fetcher = DataFetcherPro("https://example.com")

# Get all links on page
links = fetcher.extract_links(absolute=True, unique=True)

# Extract links from specific section
product_links = fetcher.extract_links(
    selector="div.products",
    unique=True
)

# Get page metadata
meta = fetcher.extract_metadata()
print(meta["title"], meta["description"])
```

#### Extract Text Content

```python
# Get all visible text
text = fetcher.fetch_text(collapse_whitespace=True)

# Extract text from specific element
article_text = fetcher.fetch_text(
    selector="article.main-content"
)
```

#### JavaScript-Rendered Pages (Requires Playwright)

```python
# Install playwright first: pip install playwright && playwright install

fetcher = DataFetcherPro("https://example.com/spa-page")
html = fetcher.render_js_page(
    wait_for=3.0,
    selector="div.loaded-content"
)
```

### API Consumption

```python
# GET request with parameters
fetcher = DataFetcherPro("https://api.example.com/users")
df = fetcher.fetch_api_json(params={"page": 1, "limit": 100})

# POST request with JSON body
fetcher = DataFetcherPro("https://api.example.com/search")
df = fetcher.fetch_api_json(
    method="POST",
    json_body={"query": "data science", "filters": ["recent"]}
)
```

### Rate Limiting and Robots.txt

```python
# Respect rate limits and robots.txt
fetcher = DataFetcherPro(
    "https://example.com/data.csv",
    request_delay=1.0  # Wait 1 second between requests
)

# Robots.txt is automatically checked for scraping operations
```

### DataFrame Operations

```python
# Append multiple datasets
fetcher1 = DataFetcherPro("https://example.com/data1.csv")
fetcher2 = DataFetcherPro("https://example.com/data2.csv")

df1 = fetcher1.fetch()
df2 = fetcher2.fetch()

fetcher1.append(fetcher2)  # In-place append
combined_df = fetcher1.df

# Merge datasets
merged = fetcher1.merge(fetcher2, on="id", how="inner")

# Save in different format
fetcher1.save("output.parquet", fmt="parquet")
fetcher1.save("output.xlsx", fmt="excel")
```

## API Reference

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | str | required | Source URL (http/https) |
| `filename` | str | None | Local filename (auto-derived if None) |
| `suppress_warnings` | bool | True | Suppress Python warnings |
| `headers` | dict | None | Custom HTTP headers |
| `auth` | tuple/AuthBase | None | Authentication credentials |
| `timeout` | int | 30 | Request timeout (seconds) |
| `request_delay` | float | 0.0 | Delay between requests (seconds) |
| `max_retries` | int | 3 | Number of retry attempts |
| `backoff_factor` | float | 0.3 | Retry backoff multiplier |

### Key Methods

#### `download(save=True, expected_size=None, expected_hash=None, hash_algo='md5', verify_ssl=True, stream=True, extract=False)`
Download the resource with optional validation and extraction.

#### `load(from_buffer=False, chunksize=None, delimiter=None, encoding=None, sheet_name=0, dtype=None, parse_dates=False, **read_kwargs)`
Load file into DataFrame with format-specific parameters.

#### `fetch(save=True, **download_kwargs)`
Convenience method: download + load in one call.

#### `fetch_html_table(url=None, table_index=0, attrs=None, header=0)`
Extract HTML table as DataFrame.

#### `fetch_api_json(params=None, method='GET', json_body=None)`
Call JSON API and return DataFrame.

#### `save(out_path, fmt=None, **save_kwargs)`
Save DataFrame to disk in specified format.

#### `summary(n_head=5)`
Print DataFrame info, head, and describe.

## Supported File Formats

| Format | Extension | Read | Write | Notes |
|--------|-----------|------|-------|-------|
| CSV | .csv | ✓ | ✓ | Chunked reading supported |
| Excel | .xlsx, .xls | ✓ | ✓ | Requires openpyxl/xlrd |
| JSON | .json | ✓ | ✓ | Auto-normalizes nested structures |
| XML | .xml | ✓ | ✓ | Fallback parser included |
| Parquet | .parquet | ✓ | ✓ | Requires pyarrow/fastparquet |
| Feather | .feather | ✓ | ✓ | Requires pyarrow |
| HDF5 | .h5, .hdf5 | ✓ | ✓ | Requires tables |
| Stata | .dta | ✓ | ✓ | |
| SAS | .sas7bdat | ✓ | ✗ | |
| Pickle | .pkl | ✓ | ✓ | Security warning applies |

## Use Cases

### ETL Pipelines
```python
# Extract
fetcher = DataFetcherPro("https://source.com/raw_data.csv")
df = fetcher.fetch()

# Transform
df = df[df['value'] > 0].reset_index(drop=True)

# Load
fetcher.df = df
fetcher.save("clean_data.parquet")
```

### Multi-Source Data Consolidation
```python
sources = [
    "https://api1.com/data.json",
    "https://api2.com/data.json",
    "https://api3.com/data.json"
]

fetchers = [DataFetcherPro(url) for url in sources]
dfs = [f.fetch_api_json() for f in fetchers]

consolidated = pd.concat(dfs, ignore_index=True)
```

### Automated Reporting
```python
fetcher = DataFetcherPro(
    "https://reporting.example.com/daily_stats.xlsx",
    headers={"API-Key": "your-key"}
)

df = fetcher.fetch()
fetcher.summary()  # Quick report
fetcher.save(f"reports/report_{pd.Timestamp.now().date()}.csv")
```

## Best Practices

1. **Use `save=False` for temporary data** to avoid cluttering disk
2. **Specify `dtype` for large CSVs** to reduce memory usage
3. **Enable `extract=True`** when downloading archives
4. **Set `request_delay`** when scraping to be respectful
5. **Validate downloads** with `expected_hash` for critical data
6. **Use `chunksize`** for files larger than available RAM

## Troubleshooting

### SSL Certificate Errors
```python
# Disable SSL verification (not recommended for production)
fetcher.download(verify_ssl=False)
```

### Large File Memory Errors
```python
# Use chunked reading
df = fetcher.load(chunksize=50000, low_memory=True)
```

### Rate Limiting / 429 Errors
```python
# Increase retry backoff
fetcher = DataFetcherPro(
    url,
    max_retries=5,
    backoff_factor=2.0,
    request_delay=1.0
)
```

### Encoding Issues
```python
# Specify encoding explicitly
df = fetcher.load(encoding='latin-1')
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details.

## Changelog

### v0.1.0 (Initial Release)
- Core download/load functionality
- Archive extraction support
- Web scraping helpers
- API consumption tools
- DataFrame operations
