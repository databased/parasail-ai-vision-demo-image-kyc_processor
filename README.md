# Parasail.io Vision Demo: Image KYC Processor

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository demonstrates how to use [Parasail.io](https://parasail.io/) for AI vision model inference to extract structured **KYC (Know Your Customer)** data from identity document images. It processes passports, driver's licenses, and ID cards — extracting fields such as name, address, date of birth, ID number, and expiration date — and outputs validated JSON.

Parasail.io provides an OpenAI-compatible API for accessing powerful vision models like Qwen 2.5 VL, making it straightforward to integrate into existing workflows.

> **Part of a series**: See also the [Fireworks.ai version](https://github.com/databased/fireworks-ai-vision-demo-image-kyc_processor) and the [Multi-Provider version](https://github.com/databased/kyc-ai-vision-multi-provider) of this processor.

## Features

- **Vision-Powered Extraction**: Uses Parasail.io-hosted vision models (Qwen 2.5 VL 72B) to parse identity document images into structured JSON
- **Batch Processing**: Process entire directories of documents with parallel execution support
- **CLI Interface**: Full command-line interface built with Click for single-document and batch operations
- **Structured Output**: Pydantic-validated data models with confidence scoring and field completion analytics
- **Retry Logic**: Configurable retry with exponential backoff for API resilience
- **Multiple Output Formats**: JSON (individual + batch summary), CSV export for spreadsheet analysis
- **Automatic .env Discovery**: Searches current and parent directories for environment configuration

## Prerequisites

- Python 3.8 or higher
- A [Parasail.io](https://parasail.io/) account and API key
- Sample identity document images (placed in the `documents/` directory)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/databased/parasail-ai-vision-demo-image-kyc_processor.git
   cd parasail-ai-vision-demo-image-kyc_processor
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   # venv\Scripts\activate   # On Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install openai click pydantic python-dotenv Pillow
   ```

4. **Configure Your API Key**:
   Create a `.env` file in the project root:
   ```
   PARASAIL_API_KEY=your_parasail_api_key_here
   ```
   Replace with your actual key from [Parasail.io](https://parasail.io/).

## Usage

The processor is operated via command-line interface:

### List Available Documents
```bash
python kyc_processor.py list-documents
```

### Test API Connection
```bash
python kyc_processor.py test-connection
```

### Process a Single Document
```bash
python kyc_processor.py process-single "License-2.jpg"
```

### Process All Documents
```bash
python kyc_processor.py process-all
```

### Process All Documents in Parallel
```bash
python kyc_processor.py process-all --parallel
```

### Clear Output Files
```bash
python kyc_processor.py clear-outputs
```

## Example Output

```json
{
  "document_type": "driver_license",
  "filename": "License-2.jpg",
  "full_name": "Jane Smith",
  "date_of_birth": "03/15/1985",
  "document_number": "D1234567",
  "expiration_date": "03/15/2028",
  "address": "456 Oak Avenue",
  "city": "Springfield",
  "state_province": "IL",
  "confidence_score": 0.95,
  "processing_time_seconds": 3.42
}
```

## Project Structure

```
parasail-ai-vision-demo-image-kyc_processor/
├── kyc_processor.py       # Main application (CLI, models, processing engine)
├── documents/             # Place identity document images here
├── outputs/               # Generated results (JSON, CSV)
│   └── individual/        # Per-document extraction results
├── .env                   # Your API key (not committed)
├── .gitignore
└── README.md
```

## Configuration

Key settings are in the `Config` class at the top of `kyc_processor.py`:

| Setting | Default | Description |
|---|---|---|
| `PARASAIL_BASE_URL` | `https://api.parasail.io/v1` | Parasail API endpoint |
| `VISION_MODEL` | `Qwen/Qwen2.5-VL-72B-Instruct` | Vision model for document analysis |
| `DOCUMENTS_DIR` | `documents/` | Input directory for images |
| `OUTPUT_DIR` | `outputs/` | Output directory for results |
| `MAX_RETRIES` | `3` | API retry attempts |
| `RETRY_DELAY` | `2` | Seconds between retries |

Alternative models available on Parasail:
- `Qwen/Qwen2-VL-72B-Instruct`
- `meta-llama/Llama-3.3-70B-Instruct`

## How It Works

1. **Document Discovery**: Scans the `documents/` directory for supported image formats (.png, .jpg, .jpeg, .bmp, .tiff)
2. **Image Encoding**: Converts images to base64 for API transmission
3. **Vision Analysis**: Sends each image to Parasail.io's vision model with a structured extraction prompt
4. **JSON Parsing**: Extracts structured fields from the model's response
5. **Validation**: Uses Pydantic models to validate and type-check extracted data
6. **Reporting**: Generates individual results, batch summaries, and CSV exports

## Architecture

The application uses a modular class-based design:

- **`Config`** — Centralized configuration with environment variable support
- **`ExtractedData`** — Pydantic model for validated KYC field extraction
- **`DocumentLoader`** — File discovery, validation, and base64 encoding
- **`ParasailAPIClient`** — API connectivity and health checking
- **`VisionProcessor`** — Core extraction engine with retry logic
- **`BatchProcessor`** — Orchestrates batch processing with parallel support and reporting

## Contributing

Contributions are welcome! Please:
1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Parasail.io](https://parasail.io/) for OpenAI-compatible vision model hosting
- Built by [databased](https://github.com/databased)
