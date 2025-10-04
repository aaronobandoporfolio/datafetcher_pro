"""
Example Usage Files for DataFetcherPro
Save these in the examples/ directory
"""

# ============================================================================
# examples
# ============================================================================
"""
Basic usage examples for DataFetcherPro
"""

from datafetcher_pro import DataFetcherPro

def example_1_simple_csv():
    """Download and load a CSV file"""
    print("=== Example 1: Simple CSV Download ===")
    
    fetcher = DataFetcherPro("https://example.com/sales_data.csv")
    df = fetcher.fetch()
    
    print(f"Loaded {len(df)} rows")
    print(df.head())


def example_2_excel_with_sheets():
    """Work with Excel files and specific sheets"""
    print("\n=== Example 2: Excel with Multiple Sheets ===")
    
    fetcher = DataFetcherPro("https://example.com/financial_report.xlsx")
    fetcher.download(save=True)
    
    # Load different sheets
    q1_data = fetcher.load(sheet_name="Q1")
    q2_data = fetcher.load(sheet_name="Q2")
    
    print(f"Q1 rows: {len(q1_data)}, Q2 rows: {len(q2_data)}")


def example_3_memory_only():
    """Work without saving to disk"""
    print("\n=== Example 3: Memory-Only Mode ===")
    
    fetcher = DataFetcherPro("https://api.example.com/temp_data.json")
    df = fetcher.fetch(save=False)
    
    # Process and save only the result
    df_filtered = df[df['score'] > 80]
    fetcher.df = df_filtered
    fetcher.save("high_scores.csv")


def example_4_with_authentication():
    """Download from authenticated endpoints"""
    print("\n=== Example 4: With Authentication ===")
    
    # API token
    fetcher = DataFetcherPro(
        "https://api.example.com/protected/data.json",
        headers={"Authorization": "Bearer YOUR_API_TOKEN"}
    )
    df = fetcher.fetch()
    
    # Basic auth
    fetcher2 = DataFetcherPro(
        "https://secure.example.com/data.csv",
        auth=("username", "password")
    )
    df2 = fetcher2.fetch()


def example_5_large_csv_chunked():
    """Handle large CSV files efficiently"""
    print("\n=== Example 5: Large CSV with Chunking ===")
    
    fetcher = DataFetcherPro("https://example.com/huge_dataset.csv")
    fetcher.download()
    
    # Load in chunks (automatically concatenates)
    df = fetcher.load(
        chunksize=10000,
        dtype={'id': int, 'value': float},
        parse_dates=['timestamp']
    )
    
    print(f"Loaded {len(df)} rows efficiently")


def example_6_archive_extraction():
    """Auto-extract ZIP, TAR, GZ files"""
    print("\n=== Example 6: Archive Extraction ===")
    
    fetcher = DataFetcherPro("https://example.com/dataset.zip")
    fetcher.download(extract=True)
    
    print("Extracted files:")
    for path in fetcher.extracted_paths:
        print(f"  - {path}")
    
    # Load one of the extracted files
    if fetcher.extracted_paths:
        first_file = fetcher.extracted_paths[0]
        # Create new fetcher for the extracted file
        # (or manually load with pandas)


def example_7_data_validation():
    """Verify file integrity with hash"""
    print("\n=== Example 7: Data Validation ===")
    
    fetcher = DataFetcherPro("https://example.com/critical_data.csv")
    
    try:
        df = fetcher.fetch(
            expected_hash="5d41402abc4b2a76b9719d911017c592",
            hash_algo="md5"
        )
        print("✓ File integrity verified")
    except RuntimeError as e:
        print(f"✗ Validation failed: {e}")


if __name__ == "__main__":
    # Run examples (comment out as needed)
    example_1_simple_csv()
    # example_2_excel_with_sheets()
    # example_3_memory_only()
    # example_4_with_authentication()
    # example_5_large_csv_chunked()
    # example_6_archive_extraction()
    # example_7_data_validation()


# ============================================================================
# examples/web_scraping.py
# ============================================================================
"""
Web scraping examples with DataFetcherPro
"""

from datafetcher_pro import DataFetcherPro
import pandas as pd


def example_scrape_tables():
    """Extract tables from HTML pages"""
    print("=== Scraping HTML Tables ===")
    
    fetcher = DataFetcherPro("https://example.com/statistics.html")
    
    # Get all tables on the page
    tables = fetcher.fetch_html_tables()
    print(f"Found {len(tables)} tables")
    
    # Get specific table by index
    main_table = fetcher.fetch_html_table(table_index=0)
    print(main_table.head())
    
    # Select table by CSS selector
    specific_table = fetcher.fetch_html_table_by_selector(
        selector="table.data-table",
        table_index=0
    )


def example_extract_links():
    """Extract and filter links from pages"""
    print("\n=== Extracting Links ===")
    
    fetcher = DataFetcherPro("https://example.com/resources")
    
    # Get all unique links (absolute URLs)
    all_links = fetcher.extract_links(unique=True, absolute=True)
    print(f"Total links: {len(all_links)}")
    
    # Extract from specific section
    article_links = fetcher.extract_links(
        selector="div.articles",
        unique=True,
        absolute=True
    )
    
    # Filter for PDFs
    pdf_links = [link for link in all_links if link.endswith('.pdf')]
    print(f"PDF links: {len(pdf_links)}")


def example_extract_metadata():
    """Get page metadata"""
    print("\n=== Extracting Metadata ===")
    
    fetcher = DataFetcherPro("https://example.com/article")
    meta = fetcher.extract_metadata()
    
    print(f"Title: {meta['title']}")
    print(f"Description: {meta['description']}")
    print(f"Keywords: {meta['keywords']}")
    print(f"Image: {meta['og:image']}")


def example_extract_text():
    """Extract clean text content"""
    print("\n=== Extracting Text ===")
    
    fetcher = DataFetcherPro("https://example.com/blog-post")
    
    # Get all visible text
    full_text = fetcher.fetch_text(collapse_whitespace=True)
    
    # Extract from specific element
    article_text = fetcher.fetch_text(
        selector="article.main-content",
        collapse_whitespace=True
    )
    
    print(f"Article length: {len(article_text)} characters")


def example_scrape_with_rate_limiting():
    """Scrape multiple pages responsibly"""
    print("\n=== Scraping with Rate Limiting ===")
    
    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
    ]
    
    # Configure rate limiting
    fetcher = DataFetcherPro(
        urls[0],
        request_delay=1.0  # Wait 1 second between requests
    )
    
    all_tables = []
    for url in urls:
        try:
            table = fetcher.fetch_html_table(url=url, table_index=0)
            all_tables.append(table)
            print(f"✓ Scraped {url}")
        except Exception as e:
            print(f"✗ Failed {url}: {e}")
    
    # Combine all tables
    if all_tables:
        combined = pd.concat(all_tables, ignore_index=True)
        print(f"Total rows collected: {len(combined)}")


def example_render_javascript():
    """Scrape JavaScript-rendered pages (requires Playwright)"""
    print("\n=== Rendering JavaScript Pages ===")
    
    try:
        fetcher = DataFetcherPro("https://example.com/spa-app")
        
        # Render the page and wait for content
        html = fetcher.render_js_page(
            wait_for=3.0,
            selector="div.dynamic-content"
        )
        
        # Parse the rendered HTML
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'lxml')
        # ... extract data from soup
        
    except RuntimeError as e:
        print(f"Playwright not installed: {e}")


if __name__ == "__main__":
    # example_scrape_tables()
    # example_extract_links()
    # example_extract_metadata()
    # example_extract_text()
    # example_scrape_with_rate_limiting()
    # example_render_javascript()
    pass


# ============================================================================
# examples/api_consumption.py
# ============================================================================
"""
API consumption examples with DataFetcherPro
"""

from datafetcher_pro import DataFetcherPro
import pandas as pd


def example_simple_api():
    """Basic GET request to JSON API"""
    print("=== Simple API Call ===")
    
    fetcher = DataFetcherPro("https://api.example.com/users")
    df = fetcher.fetch_api_json()
    
    print(f"Fetched {len(df)} users")
    print(df.head())


def example_api_with_params():
    """API call with query parameters"""
    print("\n=== API with Parameters ===")
    
    fetcher = DataFetcherPro("https://api.example.com/products")
    df = fetcher.fetch_api_json(params={
        'category': 'electronics',
        'limit': 100,
        'sort': 'price'
    })
    
    print(df.head())


def example_api_post():
    """POST request to API"""
    print("\n=== API POST Request ===")
    
    fetcher = DataFetcherPro("https://api.example.com/search")
    df = fetcher.fetch_api_json(
        method="POST",
        json_body={
            'query': 'data science',
            'filters': {
                'date_from': '2024-01-01',
                'category': 'analytics'
            }
        }
    )
    
    print(f"Search results: {len(df)} items")


def example_authenticated_api():
    """API with authentication"""
    print("\n=== Authenticated API ===")
    
    # Bearer token
    fetcher = DataFetcherPro(
        "https://api.example.com/private/data",
        headers={
            "Authorization": "Bearer YOUR_ACCESS_TOKEN",
            "Content-Type": "application/json"
        }
    )
    df = fetcher.fetch_api_json()
    
    # API Key in header
    fetcher2 = DataFetcherPro(
        "https://api.example.com/data",
        headers={"X-API-Key": "your-api-key-here"}
    )
    df2 = fetcher2.fetch_api_json()


def example_paginated_api():
    """Handle paginated API responses"""
    print("\n=== Paginated API ===")
    
    base_url = "https://api.example.com/items"
    all_data = []
    page = 1
    
    while True:
        fetcher = DataFetcherPro(base_url)
        df = fetcher.fetch_api_json(params={'page': page, 'per_page': 100})
        
        if df.empty:
            break
            
        all_data.append(df)
        print(f"Fetched page {page}: {len(df)} items")
        page += 1
        
        # Stop if we got less than full page (last page)
        if len(df) < 100:
            break
    
    # Combine all pages
    full_dataset = pd.concat(all_data, ignore_index=True)
    print(f"Total items: {len(full_dataset)}")


def example_api_error_handling():
    """Handle API errors gracefully"""
    print("\n=== API Error Handling ===")
    
    endpoints = [
        "https://api.example.com/data1",
        "https://api.example.com/data2",
        "https://api.example.com/data3",
    ]
    
    results = []
    for endpoint in endpoints:
        try:
            fetcher = DataFetcherPro(endpoint, timeout=10)
            df = fetcher.fetch_api_json()
            results.append(df)
            print(f"✓ Success: {endpoint}")
        except Exception as e:
            print(f"✗ Failed: {endpoint} - {e}")
            continue
    
    if results:
        combined = pd.concat(results, ignore_index=True)
        print(f"Successfully fetched {len(results)} endpoints")


def example_api_rate_limiting():
    """Respect API rate limits"""
    print("\n=== API with Rate Limiting ===")
    
    fetcher = DataFetcherPro(
        "https://api.example.com/data",
        request_delay=1.0,  # 1 second between requests
        max_retries=5,
        backoff_factor=2.0
    )
    
    # Make multiple requests
    ids = [1, 2, 3, 4, 5]
    data = []
    
    for item_id in ids:
        df = fetcher.fetch_api_json(params={'id': item_id})
        data.append(df)
    
    combined = pd.concat(data, ignore_index=True)


if __name__ == "__main__":
    # example_simple_api()
    # example_api_with_params()
    # example_api_post()
    # example_authenticated_api()
    # example_paginated_api()
    # example_api_error_handling()
    # example_api_rate_limiting()
    pass


# ============================================================================
# examples/etl_pipeline.py
# ============================================================================
"""
ETL pipeline examples with DataFetcherPro
"""

from datafetcher_pro import DataFetcherPro
import pandas as pd
from datetime import datetime


def example_simple_etl():
    """Simple Extract-Transform-Load pipeline"""
    print("=== Simple ETL Pipeline ===")
    
    # EXTRACT
    print("Extracting data...")
    fetcher = DataFetcherPro("https://example.com/raw_sales.csv")
    df = fetcher.fetch()
    
    # TRANSFORM
    print("Transforming data...")
    # Clean column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Filter invalid records
    df = df[df['amount'] > 0]
    
    # Add calculated columns
    df['total_with_tax'] = df['amount'] * 1.08
    df['processed_date'] = datetime.now()
    
    # LOAD
    print("Loading data...")
    fetcher.df = df
    fetcher.save("cleaned_sales.parquet", fmt="parquet")
    fetcher.save("cleaned_sales.xlsx", fmt="excel")
    
    print(f"✓ Processed {len(df)} records")


def example_multi_source_etl():
    """Combine data from multiple sources"""
    print("\n=== Multi-Source ETL ===")
    
    # EXTRACT from multiple sources
    print("Extracting from multiple sources...")
    
    # Source 1: Sales data
    sales_fetcher = DataFetcherPro("https://example.com/sales.csv")
    sales_df = sales_fetcher.fetch()
    
    # Source 2: Customer data
    customers_fetcher = DataFetcherPro("https://example.com/customers.json")
    customers_df = customers_fetcher.fetch()
    
    # Source 3: Products from API
    products_fetcher = DataFetcherPro("https://api.example.com/products")
    products_df = products_fetcher.fetch_api_json()
    
    # TRANSFORM
    print("Transforming and merging...")
    
    # Merge sales with customers
    enriched = pd.merge(
        sales_df,
        customers_df,
        left_on='customer_id',
        right_on='id',
        how='left'
    )
    
    # Add product information
    enriched = pd.merge(
        enriched,
        products_df,
        left_on='product_id',
        right_on='id',
        how='left',
        suffixes=('', '_product')
    )
    
    # Clean and aggregate
    enriched['revenue'] = enriched['quantity'] * enriched['price']
    summary = enriched.groupby('customer_id').agg({
        'revenue': 'sum',
        'quantity': 'sum',
        'order_id': 'count'
    }).reset_index()
    
    # LOAD
    print("Loading final dataset...")
    output_fetcher = DataFetcherPro("https://example.com/dummy")
    output_fetcher.df = summary
    output_fetcher.save("customer_summary.csv")
    
    print(f"✓ Created summary for {len(summary)} customers")


def example_incremental_etl():
    """Incremental data loading"""
    print("\n=== Incremental ETL ===")
    
    # Load existing data
    try:
        existing_df = pd.read_parquet("data_warehouse.parquet")
        last_date = existing_df['date'].max()
        print(f"Last processed date: {last_date}")
    except FileNotFoundError:
        existing_df = pd.DataFrame()
        last_date = None
    
    # EXTRACT new data only
    fetcher = DataFetcherPro("https://api.example.com/transactions")
    if last_date:
        new_df = fetcher.fetch_api_json(params={'since': last_date.isoformat()})
    else:
        new_df = fetcher.fetch_api_json()
    
    print(f"Found {len(new_df)} new records")
    
    # TRANSFORM
    new_df['processed_at'] = datetime.now()
    
    # LOAD - Append to existing
    if not existing_df.empty:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    
    # Remove duplicates based on ID
    combined_df = combined_df.drop_duplicates(subset=['transaction_id'], keep='last')
    
    # Save back
    combined_df.to_parquet("data_warehouse.parquet", index=False)
    print(f"✓ Total records in warehouse: {len(combined_df)}")


def example_data_quality_checks():
    """ETL with data quality validation"""
    print("\n=== ETL with Quality Checks ===")
    
    # EXTRACT
    fetcher = DataFetcherPro("https://example.com/survey_data.csv")
    df = fetcher.fetch()
    
    initial_count = len(df)
    print(f"Initial records: {initial_count}")
    
    # QUALITY CHECKS
    issues = []
    
    # Check 1: Missing critical fields
    required_fields = ['id', 'response', 'date']
    for field in required_fields:
        missing = df[field].isna().sum()
        if missing > 0:
            issues.append(f"Missing {field}: {missing} records")
            df = df.dropna(subset=[field])
    
    # Check 2: Data type validation
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        issues.append(f"Date parsing errors: {e}")
    
    # Check 3: Value ranges
    if 'score' in df.columns:
        invalid_scores = df[(df['score'] < 0) | (df['score'] > 100)]
        if len(invalid_scores) > 0:
            issues.append(f"Invalid scores: {len(invalid_scores)} records")
            df = df[(df['score'] >= 0) & (df['score'] <= 100)]
    
    # Check 4: Duplicates
    duplicates = df.duplicated(subset=['id']).sum()
    if duplicates > 0:
        issues.append(f"Duplicate IDs: {duplicates} records")
        df = df.drop_duplicates(subset=['id'], keep='first')
    
    # Report issues
    print("\nQuality Check Results:")
    if issues:
        for issue in issues:
            print(f"  ⚠ {issue}")
    else:
        print("  ✓ All checks passed")
    
    final_count = len(df)
    print(f"\nRecords after cleaning: {final_count}")
    print(f"Records removed: {initial_count - final_count}")
    
    # LOAD with metadata
    df['quality_checked'] = True
    df['check_date'] = datetime.now()
    
    fetcher.df = df
    fetcher.save("quality_checked_data.parquet")
    
    # Save quality report
    report = pd.DataFrame({
        'check_date': [datetime.now()],
        'initial_records': [initial_count],
        'final_records': [final_count],
        'issues_found': [len(issues)],
        'issues_detail': ['; '.join(issues)]
    })
    report.to_csv("quality_report.csv", index=False)


def example_scheduled_etl():
    """ETL pipeline suitable for scheduling"""
    print("\n=== Scheduled ETL Pipeline ===")
    
    try:
        # Configuration
        source_url = "https://api.example.com/daily_metrics"
        output_path = f"metrics_{datetime.now().strftime('%Y%m%d')}.parquet"
        
        # EXTRACT
        print(f"[{datetime.now()}] Starting extraction...")
        fetcher = DataFetcherPro(
            source_url,
            headers={"Authorization": "Bearer TOKEN"},
            timeout=60
        )
        df = fetcher.fetch_api_json()
        print(f"✓ Extracted {len(df)} records")
        
        # TRANSFORM
        print(f"[{datetime.now()}] Transforming...")
        df['etl_date'] = datetime.now().date()
        df['etl_timestamp'] = datetime.now()
        
        # Business logic
        df['metric_category'] = df['value'].apply(
            lambda x: 'high' if x > 100 else 'medium' if x > 50 else 'low'
        )
        
        # LOAD
        print(f"[{datetime.now()}] Loading...")
        fetcher.df = df
        fetcher.save(output_path)
        
        # Log success
        print(f"✓ ETL completed successfully at {datetime.now()}")
        print(f"✓ Output saved to: {output_path}")
        
        # Optional: Send success notification
        # send_notification("ETL Success", f"Processed {len(df)} records")
        
    except Exception as e:
        # Log error
        print(f"✗ ETL failed at {datetime.now()}: {e}")
        # Optional: Send alert
        # send_alert("ETL Failed", str(e))
        raise


def example_parallel_etl():
    """Process multiple datasets in parallel (conceptual)"""
    print("\n=== Parallel ETL Processing ===")
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    sources = [
        "https://api.example.com/dataset1",
        "https://api.example.com/dataset2",
        "https://api.example.com/dataset3",
        "https://api.example.com/dataset4",
    ]
    
    def process_source(url):
        """Process a single data source"""
        try:
            fetcher = DataFetcherPro(url, timeout=30)
            df = fetcher.fetch_api_json()
            
            # Transform
            df['source'] = url
            df['processed_at'] = datetime.now()
            
            return df
        except Exception as e:
            print(f"✗ Failed to process {url}: {e}")
            return pd.DataFrame()
    
    # Process in parallel
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_source, url): url for url in sources}
        
        for future in as_completed(futures):
            url = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    results.append(df)
                    print(f"✓ Processed {url}: {len(df)} records")
            except Exception as e:
                print(f"✗ Error processing {url}: {e}")
    
    # Combine results
    if results:
        final_df = pd.concat(results, ignore_index=True)
        print(f"\n✓ Total records processed: {len(final_df)}")
        final_df.to_parquet("combined_parallel_output.parquet")


if __name__ == "__main__":
    # example_simple_etl()
    # example_multi_source_etl()
    # example_incremental_etl()
    # example_data_quality_checks()
    # example_scheduled_etl()
    # example_parallel_etl()
    pass
