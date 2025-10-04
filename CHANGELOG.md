# Changelog

All notable changes to DataFetcherPro will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Async download support
- Database output support (SQL, MongoDB)
- Progress bars for large downloads
- Caching mechanism

## [0.1.0] - 2025-01-15

### Added
- Initial release
- Core download functionality with retry logic
- Support for CSV, Excel, JSON, XML, Parquet, Feather, HDF5, Stata, SAS, Pickle
- Archive extraction (ZIP, TAR, GZ)
- Web scraping helpers (HTML tables, links, metadata, text)
- API consumption tools
- DataFrame operations (append, merge, save)
- Robots.txt respect and rate limiting
- Hash validation for downloads
- Chunked CSV reading
- Memory-only mode (no disk write)
- Comprehensive test suite
- Full documentation

### Security
- Added SSL verification option
- Input validation for URLs

## [0.0.1] - 2024-12-01

### Added
- Project structure
- Basic prototype
