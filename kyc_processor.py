"""
FSI KYC Identity Verification PoC using Parasail.io
A comprehensive identity document processing system for financial services compliance.

This implementation provides automated extraction of identity information from 
driver's licenses, passports, and ID cards using vision AI models on Parasail.io.
"""

import os
import json
import base64
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
from pydantic import BaseModel, ValidationError, Field
from openai import OpenAI  # Parasail.io uses OpenAI-compatible API
from PIL import Image
from dotenv import load_dotenv, find_dotenv

# Configure logging with both file and console output
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kyc_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =====================================================
# ENVIRONMENT SETUP - Automatic .env Discovery
# =====================================================

# Automatically search for .env file in current and parent directories
env_file = find_dotenv()
if env_file:
    logger.info(f"Found .env file at: {env_file}")
    load_dotenv(env_file)
else:
    logger.warning("No .env file found in current or parent directories")

# =====================================================
# CONFIGURATION
# =====================================================

class Config:
    """
    Central configuration for the KYC processing system using Parasail.io.
    
    All key parameters are defined here for easy modification and deployment.
    Environment variables take precedence over default values.
    """
    # API Configuration for Parasail.io
    PARASAIL_API_KEY = os.getenv("PARASAIL_API_KEY")
    if not PARASAIL_API_KEY:
        logger.warning("PARASAIL_API_KEY not found in environment variables")
    
    PARASAIL_BASE_URL = "https://api.parasail.io/v1"  # Parasail API endpoint
    
    # Directory Configuration
    DOCUMENTS_DIR = Path("documents")  # Input directory for identity documents
    OUTPUT_DIR = Path("outputs")  # Main output directory
    INDIVIDUAL_OUTPUT_DIR = OUTPUT_DIR / "individual"  # Individual document results
    
    # Supported image formats for identity documents
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    # Vision model configuration - Using Parasail's available models
    # Based on your available models, using Qwen 2.5 VL for vision tasks
    VISION_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"  # Vision-Language model available on Parasail
    
    # Alternative model options based on your Parasail environment:
    # VISION_MODEL = "parasail-llama-33-70b-fp8"  # For general purpose
    # VISION_MODEL = "meta-llama/Llama-3.3-70B-Instruct"  # Alternative Llama model
    
    # API retry configuration for resilience
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds between retries

# Initialize output directories on module load
Config.OUTPUT_DIR.mkdir(exist_ok=True)
Config.INDIVIDUAL_OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize Parasail API client with configuration
if Config.PARASAIL_API_KEY:
    client = OpenAI(
        api_key=Config.PARASAIL_API_KEY,
        base_url=Config.PARASAIL_BASE_URL,
    )
else:
    client = None
    logger.error("Parasail client not initialized - API key missing")

# =====================================================
# DATA MODELS
# =====================================================

class ExtractedData(BaseModel):
    """
    Comprehensive data model for extracted identity document information.
    
    This model represents all possible fields that might be extracted from
    various identity documents (driver's licenses, passports, ID cards).
    Fields are optional to handle different document types gracefully.
    """
    # Document metadata
    document_type: str = Field(description="Type of document: driver_license, passport, id_card")
    filename: str = Field(description="Original filename of the processed document")
    
    # Personal Information
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    middle_name: Optional[str] = None
    
    # Important dates
    date_of_birth: Optional[str] = None
    expiration_date: Optional[str] = None
    issue_date: Optional[str] = None
    
    # Document identifiers
    document_number: Optional[str] = None
    document_class: Optional[str] = None  # e.g., Class A, B, C for licenses
    
    # Address Information
    address: Optional[str] = None
    city: Optional[str] = None
    state_province: Optional[str] = None
    zip_postal_code: Optional[str] = None
    country: Optional[str] = None
    
    # Physical Characteristics (common on licenses)
    sex_gender: Optional[str] = None
    height: Optional[str] = None
    weight: Optional[str] = None
    eye_color: Optional[str] = None
    hair_color: Optional[str] = None
    
    # Document Features
    has_photo: bool = False
    issuing_authority: Optional[str] = None
    restrictions: Optional[str] = None  # Driving restrictions
    endorsements: Optional[str] = None  # Special endorsements
    
    # Processing Metadata for quality assurance
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    processing_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    processing_time_seconds: Optional[float] = None
    raw_extraction: Dict[str, Any] = Field(default_factory=dict)  # Store raw API response

@dataclass
class DocumentInfo:
    """
    Container for document file information and validation status.
    
    Used during the discovery phase to track which files can be processed.
    """
    path: Path
    filename: str
    size_mb: float
    format: str
    is_valid: bool
    error_message: Optional[str] = None

@dataclass  
class ProcessingResult:
    """
    Result container for individual document processing.
    
    Captures both successful extractions and failures with detailed error information.
    """
    success: bool
    document_info: DocumentInfo
    extracted_data: Optional[ExtractedData] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None

class BatchSummary(BaseModel):
    """
    Comprehensive summary of batch processing results.
    
    Provides analytics and metrics for monitoring processing quality and performance.
    """
    total_documents: int
    successful_extractions: int
    failed_extractions: int
    success_rate: float
    average_processing_time: float
    total_processing_time: float
    document_types: Dict[str, int]  # Distribution of document types
    field_completion_rates: Dict[str, float]  # Percentage of non-null fields
    processing_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# =====================================================
# DOCUMENT LOADING AND VALIDATION
# =====================================================

class DocumentLoader:
    """
    Handles discovery, validation, and preparation of identity documents.
    
    This class is responsible for finding valid image files in the documents
    directory and preparing them for processing by the vision model.
    """
    
    @staticmethod
    def discover_documents(directory: Path = Config.DOCUMENTS_DIR) -> List[DocumentInfo]:
        """
        Discover all valid image documents in the specified directory.
        
        Scans the directory for supported image formats and validates each file.
        
        Args:
            directory: Path to scan for documents (defaults to Config.DOCUMENTS_DIR)
            
        Returns:
            List of DocumentInfo objects with validation status
        """
        documents = []
        
        # Ensure directory exists
        if not directory.exists():
            logger.warning(f"Documents directory {directory} does not exist")
            return documents
        
        logger.info(f"Scanning directory: {directory}")
        
        # Process each file in the directory
        for file_path in directory.iterdir():
            if file_path.is_file():
                doc_info = DocumentLoader._validate_document(file_path)
                documents.append(doc_info)
                
                # Log validation results
                if doc_info.is_valid:
                    logger.info(f"✅ Valid document: {doc_info.filename} ({doc_info.size_mb:.1f}MB)")
                else:
                    logger.warning(f"❌ Invalid document: {doc_info.filename} - {doc_info.error_message}")
        
        return documents
    
    @staticmethod
    def _validate_document(file_path: Path) -> DocumentInfo:
        """
        Validate a single document file.
        
        Checks file format, size, and attempts to open as an image to ensure validity.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            DocumentInfo with validation results
        """
        try:
            # Check if file extension is supported
            if file_path.suffix.lower() not in Config.SUPPORTED_FORMATS:
                return DocumentInfo(
                    path=file_path,
                    filename=file_path.name,
                    size_mb=0,
                    format=file_path.suffix,
                    is_valid=False,
                    error_message=f"Unsupported format: {file_path.suffix}"
                )
            
            # Get file size in MB
            size_bytes = file_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            
            # Verify the file can be opened as an image
            try:
                with Image.open(file_path) as img:
                    format_name = img.format
            except Exception as e:
                return DocumentInfo(
                    path=file_path,
                    filename=file_path.name,
                    size_mb=size_mb,
                    format=file_path.suffix,
                    is_valid=False,
                    error_message=f"Cannot open as image: {str(e)}"
                )
            
            # File is valid
            return DocumentInfo(
                path=file_path,
                filename=file_path.name,
                size_mb=size_mb,
                format=format_name or file_path.suffix,
                is_valid=True
            )
            
        except Exception as e:
            # Handle any unexpected errors
            return DocumentInfo(
                path=file_path,
                filename=file_path.name,
                size_mb=0,
                format="unknown",
                is_valid=False,
                error_message=f"Error accessing file: {str(e)}"
            )
    
    @staticmethod
    def encode_image_base64(file_path: Path) -> str:
        """
        Encode an image file to base64 string for API transmission.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Base64 encoded string of the image
            
        Raises:
            Exception: If file cannot be read or encoded
        """
        try:
            with open(file_path, 'rb') as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded
        except Exception as e:
            logger.error(f"Error encoding image {file_path}: {e}")
            raise

# =====================================================
# PARASAIL API CLIENT
# =====================================================

class ParasailAPIClient:
    """
    Client for interacting with Parasail.io API.
    
    Provides methods for testing connectivity and making API calls.
    """
    
    def __init__(self):
        if not client:
            raise ValueError("Parasail client not initialized - check PARASAIL_API_KEY")
        self.client = client
        self.model = Config.VISION_MODEL
    
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test API connection with a simple request.
        
        Verifies that the API key is valid and the service is accessible.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Make a simple completion request
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "Hello, can you confirm this API is working?"}
                ],
                max_tokens=50
            )
            
            if response.choices:
                return True, "API connection successful"
            else:
                return False, "No response from API"
                
        except Exception as e:
            return False, f"API connection failed: {str(e)}"

# =====================================================
# VISION PROCESSING ENGINE
# =====================================================

class VisionProcessor:
    """
    Core processing engine using Parasail vision models.
    
    Handles the extraction of structured data from identity document images
    using the configured vision model on Parasail.io.
    """
    
    def __init__(self):
        self.client = ParasailAPIClient()
        
    def extract_document_data(self, document_info: DocumentInfo) -> ExtractedData:
        """
        Extract structured data from a document image using vision AI.
        
        This method sends the document image to the vision model with a carefully
        crafted prompt to extract all relevant KYC information.
        
        Args:
            document_info: Information about the document to process
            
        Returns:
            ExtractedData object with all extracted fields
            
        Raises:
            ValueError: If image encoding fails
            Exception: If API calls fail after retries
        """
        
        # Encode the image for API transmission
        try:
            image_base64 = DocumentLoader.encode_image_base64(document_info.path)
        except Exception as e:
            raise ValueError(f"Failed to encode image: {e}")
        
        # Comprehensive prompt for KYC data extraction
        # This prompt is designed to extract maximum information while maintaining accuracy
        system_prompt = """You are an expert identity document analyzer for financial services KYC compliance.
        Analyze the provided identity document image and extract ALL visible information with high precision.
        
        Return ONLY a valid JSON object with this exact structure (use null for missing fields):
        {
            "document_type": "driver_license" | "passport" | "id_card",
            "full_name": "complete name as shown",
            "first_name": "first name only",
            "last_name": "last name only", 
            "middle_name": "middle name or initial",
            "date_of_birth": "MM/DD/YYYY format",
            "expiration_date": "MM/DD/YYYY format",
            "issue_date": "MM/DD/YYYY format",
            "document_number": "ID/license/passport number",
            "document_class": "license class if applicable",
            "address": "complete address",
            "city": "city name",
            "state_province": "state/province",
            "zip_postal_code": "zip or postal code",
            "country": "country name or code",
            "sex_gender": "M/F or Male/Female",
            "height": "height as shown",
            "weight": "weight as shown",
            "eye_color": "eye color",
            "hair_color": "hair color", 
            "has_photo": true/false,
            "issuing_authority": "issuing organization",
            "restrictions": "any restrictions noted",
            "endorsements": "endorsements if any",
            "confidence_score": 0.95,
            "security_features": ["visible security elements"],
            "notes": "additional observations"
        }
        
        Extract only clearly visible information. Use precise formatting for dates and be consistent."""
        
        # Attempt API call with retry logic for resilience
        for attempt in range(Config.MAX_RETRIES):
            try:
                # Make the API request to the vision model on Parasail
                response = client.chat.completions.create(
                    model=Config.VISION_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Analyze this identity document ({document_info.filename}) and extract all KYC-relevant information:"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0.1,  # Low temperature for consistent extraction
                    max_tokens=2000
                )
                
                # Parse the response
                content = response.choices[0].message.content.strip()
                raw_data = self._parse_json_response(content)
                
                # If successful, break out of retry loop
                if "error" not in raw_data:
                    break
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(Config.RETRY_DELAY)
                else:
                    raise e
        
        # Create structured data object from raw extraction
        extracted_data = ExtractedData(
            filename=document_info.filename,
            document_type=raw_data.get("document_type", "unknown"),
            full_name=raw_data.get("full_name"),
            first_name=raw_data.get("first_name"),
            last_name=raw_data.get("last_name"),
            middle_name=raw_data.get("middle_name"),
            date_of_birth=raw_data.get("date_of_birth"),
            expiration_date=raw_data.get("expiration_date"),
            issue_date=raw_data.get("issue_date"),
            document_number=raw_data.get("document_number"),
            document_class=raw_data.get("document_class"),
            address=raw_data.get("address"),
            city=raw_data.get("city"),
            state_province=raw_data.get("state_province"),
            zip_postal_code=raw_data.get("zip_postal_code"),
            country=raw_data.get("country"),
            sex_gender=raw_data.get("sex_gender"),
            height=raw_data.get("height"),
            weight=raw_data.get("weight"),
            eye_color=raw_data.get("eye_color"),
            hair_color=raw_data.get("hair_color"),
            has_photo=raw_data.get("has_photo", False),
            issuing_authority=raw_data.get("issuing_authority"),
            restrictions=raw_data.get("restrictions"),
            endorsements=raw_data.get("endorsements"),
            confidence_score=raw_data.get("confidence_score", 0.0),
            raw_extraction=raw_data
        )
        
        return extracted_data
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """
        Parse JSON from API response content.
        
        Handles various response formats including markdown code blocks.
        
        Args:
            content: Raw response content from the API
            
        Returns:
            Parsed JSON as a dictionary
        """
        try:
            # Handle responses wrapped in markdown code blocks
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            elif "{" in content:
                # Find the JSON object in the response
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                json_str = content[json_start:json_end]
            else:
                json_str = content
            
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nContent: {content[:200]}...")
            return {"error": "JSON parsing failed", "raw_content": content}

# =====================================================
# BATCH PROCESSING ENGINE
# =====================================================

class BatchProcessor:
    """
    Handles batch processing of multiple documents with comprehensive reporting.
    
    Supports both sequential and parallel processing modes, generates detailed
    analytics, and saves results in multiple formats.
    """
    
    def __init__(self):
        self.vision_processor = VisionProcessor()
        self.results: List[ProcessingResult] = []
    
    def process_single_document(self, document_info: DocumentInfo) -> ProcessingResult:
        """
        Process a single document and return the result.
        
        This method handles all aspects of processing including timing, error handling,
        and result saving.
        
        Args:
            document_info: Information about the document to process
            
        Returns:
            ProcessingResult with extraction data or error information
        """
        start_time = time.time()
        
        try:
            # Skip invalid documents
            if not document_info.is_valid:
                return ProcessingResult(
                    success=False,
                    document_info=document_info,
                    error_message=document_info.error_message
                )
            
            logger.info(f"🔍 Processing: {document_info.filename}")
            
            # Extract data using vision model
            extracted_data = self.vision_processor.extract_document_data(document_info)
            
            # Calculate and add processing time
            processing_time = time.time() - start_time
            extracted_data.processing_time_seconds = processing_time
            
            # Save individual result immediately
            self._save_individual_result(extracted_data)
            
            return ProcessingResult(
                success=True,
                document_info=document_info,
                extracted_data=extracted_data,
                processing_time=processing_time
            )
            
        except Exception as e:
            # Handle processing failures gracefully
            processing_time = time.time() - start_time
            logger.error(f"❌ Error processing {document_info.filename}: {e}")
            
            return ProcessingResult(
                success=False,
                document_info=document_info,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def process_all_documents(self, parallel: bool = False) -> BatchSummary:
        """
        Process all documents in the documents directory.
        
        This is the main entry point for batch processing. It discovers all documents,
        processes them (optionally in parallel), and generates a comprehensive report.
        
        Args:
            parallel: If True, process documents in parallel using threading
            
        Returns:
            BatchSummary with complete processing statistics
        """
        # Discover all documents in the configured directory
        documents = DocumentLoader.discover_documents()
        
        if not documents:
            logger.warning("No documents found to process")
            return BatchSummary(
                total_documents=0,
                successful_extractions=0,
                failed_extractions=0,
                success_rate=0.0,
                average_processing_time=0.0,
                total_processing_time=0.0,
                document_types={},
                field_completion_rates={}
            )
        
        logger.info(f"🚀 Starting batch processing of {len(documents)} documents (parallel={parallel})")
        
        start_time = time.time()
        
        if parallel and len(documents) > 1:
            # Parallel processing using ThreadPoolExecutor
            # Limited to 3 workers to avoid overwhelming the API
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit all valid documents for processing
                future_to_doc = {
                    executor.submit(self.process_single_document, doc): doc 
                    for doc in documents if doc.is_valid
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_doc):
                    result = future.result()
                    self.results.append(result)
        else:
            # Sequential processing
            for doc in documents:
                result = self.process_single_document(doc)
                self.results.append(result)
        
        total_time = time.time() - start_time
        
        # Generate and save comprehensive summary
        summary = self._generate_batch_summary(total_time)
        self._save_batch_summary(summary)
        
        return summary
    
    def _generate_batch_summary(self, total_time: float) -> BatchSummary:
        """
        Generate comprehensive batch processing summary with analytics.
        
        Args:
            total_time: Total time taken for batch processing
            
        Returns:
            BatchSummary with detailed statistics
        """
        # Separate successful and failed results
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        # Calculate document type distribution
        document_types = {}
        for result in successful:
            doc_type = result.extracted_data.document_type
            document_types[doc_type] = document_types.get(doc_type, 0) + 1
        
        # Calculate average processing time
        processing_times = [r.processing_time for r in successful if r.processing_time]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Calculate field completion rates
        field_completion = self._calculate_field_completion_rates(successful)
        
        return BatchSummary(
            total_documents=len(self.results),
            successful_extractions=len(successful),
            failed_extractions=len(failed),
            success_rate=len(successful) / len(self.results) * 100 if self.results else 0,
            average_processing_time=round(avg_processing_time, 2),
            total_processing_time=round(total_time, 2),
            document_types=document_types,
            field_completion_rates=field_completion
        )
    
    def _calculate_field_completion_rates(self, successful_results: List[ProcessingResult]) -> Dict[str, float]:
        """
        Calculate completion rates for each field across all documents.
        
        This helps identify which fields are consistently extracted vs problematic.
        
        Args:
            successful_results: List of successful processing results
            
        Returns:
            Dictionary mapping field names to completion percentages
        """
        if not successful_results:
            return {}
        
        field_counts = {}
        total_docs = len(successful_results)
        
        # Key fields to track for KYC compliance
        key_fields = [
            'full_name', 'first_name', 'last_name', 'date_of_birth',
            'document_number', 'expiration_date', 'address', 'state_province',
            'sex_gender', 'height', 'weight', 'eye_color', 'has_photo'
        ]
        
        # Count non-null occurrences of each field
        for field in key_fields:
            non_null_count = 0
            for result in successful_results:
                value = getattr(result.extracted_data, field, None)
                if value is not None and value != "" and value is not False:
                    non_null_count += 1
            
            field_counts[field] = round((non_null_count / total_docs) * 100, 1)
        
        return field_counts
    
    def _save_individual_result(self, extracted_data: ExtractedData):
        """
        Save individual document extraction results to a JSON file.
        
        Args:
            extracted_data: Extracted data to save
        """
        # Create filename based on original document name
        filename = f"{Path(extracted_data.filename).stem}_results.json"
        filepath = Config.INDIVIDUAL_OUTPUT_DIR / filename
        
        # Save with pretty formatting for readability
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(extracted_data.model_dump(), f, indent=2, default=str, ensure_ascii=False)
    
    def _save_batch_summary(self, summary: BatchSummary):
        """
        Save batch processing summary and detailed results.
        
        Creates multiple output files:
        - Summary JSON with statistics
        - Detailed results with all extractions
        - CSV export for spreadsheet analysis
        
        Args:
            summary: Batch summary to save
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Save summary statistics
        summary_file = Config.OUTPUT_DIR / f"batch_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary.model_dump(), f, indent=2, default=str, ensure_ascii=False)
        
        # Save detailed results including all extractions
        detailed_results = {
            "summary": summary.model_dump(),
            "successful_extractions": [
                result.extracted_data.model_dump() for result in self.results if result.success
            ],
            "failed_extractions": [
                {
                    "filename": result.document_info.filename,
                    "error": result.error_message,
                    "processing_time": result.processing_time
                }
                for result in self.results if not result.success
            ]
        }
        
        detailed_file = Config.OUTPUT_DIR / f"detailed_results_{timestamp}.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, default=str, ensure_ascii=False)
        
        # Create CSV export for easy analysis in spreadsheet applications
        self._export_to_csv(timestamp)
        
        logger.info(f"📊 Results saved to {Config.OUTPUT_DIR}")
    
    def _export_to_csv(self, timestamp: str):
        """
        Export results to CSV format for spreadsheet analysis.
        
        Args:
            timestamp: Timestamp string for filename
        """
        import csv
        
        csv_file = Config.OUTPUT_DIR / f"kyc_extractions_{timestamp}.csv"
        
        # Define CSV columns - key fields for KYC
        fieldnames = [
            'filename', 'document_type', 'full_name', 'first_name', 'last_name',
            'date_of_birth', 'document_number', 'expiration_date', 'address',
            'city', 'state_province', 'zip_postal_code', 'confidence_score',
            'processing_time_seconds'
        ]
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write successful extractions
            for result in self.results:
                if result.success:
                    row = {
                        field: getattr(result.extracted_data, field, '') 
                        for field in fieldnames
                    }
                    writer.writerow(row)

# =====================================================
# CLI INTERFACE
# =====================================================

@click.group()
def cli():
    """
    KYC Identity Verification System - Process identity documents using AI vision models on Parasail.io.
    
    This tool extracts structured data from driver's licenses, passports, and ID cards
    for financial services compliance purposes.
    """
    pass

@cli.command()
def list_documents():
    """List all documents found in the documents directory."""
    documents = DocumentLoader.discover_documents()
    
    if not documents:
        click.echo("No documents found in ./documents/")
        return
    
    click.echo(f"\nFound {len(documents)} documents in {Config.DOCUMENTS_DIR}:\n")
    
    # Display document information in a table format
    valid_count = sum(1 for d in documents if d.is_valid)
    
    for doc in documents:
        status = "✅" if doc.is_valid else "❌"
        click.echo(f"{status} {doc.filename:<30} ({doc.size_mb:.1f}MB) {doc.format}")
        if not doc.is_valid:
            click.echo(f"   └─ {doc.error_message}")
    
    click.echo(f"\nValid documents: {valid_count}/{len(documents)}")

@cli.command()
def test_connection():
    """Test connection to Parasail.io API."""
    click.echo("Testing Parasail.io API connection...")
    
    if not Config.PARASAIL_API_KEY:
        click.echo("❌ PARASAIL_API_KEY not found in environment variables")
        click.echo("Please check your .env file or set the environment variable")
        return
    
    try:
        api_client = ParasailAPIClient()
        success, message = api_client.test_connection()
        
        if success:
            click.echo(f"✅ {message}")
            click.echo(f"🚀 Model: {Config.VISION_MODEL} available on Parasail")
        else:
            click.echo(f"❌ {message}")
            click.echo("Please check your PARASAIL_API_KEY environment variable")
    except Exception as e:
        click.echo(f"❌ Error: {e}")

@cli.command()
@click.argument('filename')
@click.option('--output-format', default='json', help='Output format: json or yaml')
def process_single(filename, output_format):
    """
    Process a single document by filename.
    
    Example: python kyc_processor.py process-single License-1.png
    """
    if not Config.PARASAIL_API_KEY:
        click.echo("❌ PARASAIL_API_KEY not found. Please check your .env file")
        return
    
    # Find the document
    documents = DocumentLoader.discover_documents()
    document = None
    
    for doc in documents:
        if doc.filename == filename:
            document = doc
            break
    
    if not document:
        click.echo(f"❌ Document '{filename}' not found in {Config.DOCUMENTS_DIR}")
        return
    
    if not document.is_valid:
        click.echo(f"❌ Document '{filename}' is not valid: {document.error_message}")
        return
    
    try:
        # Process the document
        processor = BatchProcessor()
        result = processor.process_single_document(document)
        
        if result.success:
            click.echo(f"\n✅ Processing: {filename}")
            click.echo(f"📄 Document Type: {result.extracted_data.document_type}")
            
            # Count extracted fields
            extracted_fields = 0
            total_fields = 15  # Key fields we care about
            
            key_fields = ['full_name', 'date_of_birth', 'document_number', 'expiration_date',
                         'address', 'city', 'state_province', 'sex_gender']
            
            for field in key_fields:
                if getattr(result.extracted_data, field, None):
                    extracted_fields += 1
            
            click.echo(f"📊 Extracted {extracted_fields}/{len(key_fields)} key fields")
            click.echo(f"⚡ Confidence: {result.extracted_data.confidence_score:.2f}")
            click.echo(f"⏱️  Processing time: {result.processing_time:.2f} seconds")
            
            # Display key extracted information
            click.echo("\n🔍 Extracted Information:")
            if result.extracted_data.full_name:
                click.echo(f"   Name: {result.extracted_data.full_name}")
            if result.extracted_data.date_of_birth:
                click.echo(f"   DOB: {result.extracted_data.date_of_birth}")
            if result.extracted_data.document_number:
                click.echo(f"   Document #: {result.extracted_data.document_number}")
            if result.extracted_data.expiration_date:
                click.echo(f"   Expires: {result.extracted_data.expiration_date}")
            
            output_path = Config.INDIVIDUAL_OUTPUT_DIR / f"{Path(filename).stem}_results.json"
            click.echo(f"\n💾 Results saved to: {output_path}")
        else:
            click.echo(f"❌ Failed to process {filename}: {result.error_message}")
    except Exception as e:
        click.echo(f"❌ Error processing document: {e}")

@cli.command()
@click.option('--parallel', is_flag=True, help='Process documents in parallel')
@click.option('--output-format', default='json', help='Output format for results')
def process_all(parallel, output_format):
    """
    Process all documents in the documents directory.
    
    Examples:
        python kyc_processor.py process-all
        python kyc_processor.py process-all --parallel
    """
    if not Config.PARASAIL_API_KEY:
        click.echo("❌ PARASAIL_API_KEY not found. Please check your .env file")
        return
    
    click.echo("🚀 Starting batch processing on Parasail.io...")
    
    try:
        processor = BatchProcessor()
        summary = processor.process_all_documents(parallel=parallel)
        
        # Display results summary
        click.echo("\n" + "="*60)
        click.echo("📊 BATCH PROCESSING SUMMARY")
        click.echo("="*60)
        
        click.echo(f"\n📁 Total documents: {summary.total_documents}")
        click.echo(f"✅ Successful: {summary.successful_extractions}")
        click.echo(f"❌ Failed: {summary.failed_extractions}")
        click.echo(f"📈 Success rate: {summary.success_rate:.1f}%")
        
        click.echo(f"\n⏱️  Performance:")
        click.echo(f"   Average processing time: {summary.average_processing_time:.2f}s per document")
        click.echo(f"   Total processing time: {summary.total_processing_time:.2f}s")
        
        if summary.document_types:
            click.echo(f"\n📋 Document Types:")
            for doc_type, count in summary.document_types.items():
                click.echo(f"   {doc_type}: {count}")
        
        if summary.field_completion_rates:
            click.echo(f"\n📊 Field Extraction Rates:")
            for field, rate in sorted(summary.field_completion_rates.items(), 
                                     key=lambda x: x[1], reverse=True):
                bar = "█" * int(rate / 10) + "░" * (10 - int(rate / 10))
                click.echo(f"   {field:<20} {bar} {rate:.1f}%")
        
        click.echo(f"\n💾 Output files saved to: {Config.OUTPUT_DIR}")
        click.echo(f"   ├─ Summary: batch_summary_*.json")
        click.echo(f"   ├─ Details: detailed_results_*.json")
        click.echo(f"   ├─ CSV: kyc_extractions_*.csv")
        click.echo(f"   └─ Individual: {Config.INDIVIDUAL_OUTPUT_DIR}/*.json")
    except Exception as e:
        click.echo(f"❌ Error during batch processing: {e}")

@cli.command()
def clear_outputs():
    """Clear all output files (with confirmation)."""
    file_count = sum(1 for _ in Config.OUTPUT_DIR.rglob('*') if _.is_file())
    
    if file_count == 0:
        click.echo("No output files to clear.")
        return
    
    if click.confirm(f"Delete {file_count} output files?"):
        import shutil
        shutil.rmtree(Config.OUTPUT_DIR)
        Config.OUTPUT_DIR.mkdir(exist_ok=True)
        Config.INDIVIDUAL_OUTPUT_DIR.mkdir(exist_ok=True)
        click.echo(f"✅ Cleared {file_count} files")

@cli.command()
def show_config():
    """Display current configuration and environment status."""
    click.echo("\n" + "="*60)
    click.echo("⚙️  PARASAIL CONFIGURATION STATUS")
    click.echo("="*60)
    
    # Check API key
    if Config.PARASAIL_API_KEY:
        key_display = f"{Config.PARASAIL_API_KEY[:15]}..." if len(Config.PARASAIL_API_KEY) > 15 else "***"
        click.echo(f"✅ API Key: {key_display}")
    else:
        click.echo("❌ API Key: Not found")
    
    # Check .env file
    if env_file:
        click.echo(f"✅ .env File: {env_file}")
    else:
        click.echo("❌ .env File: Not found")
    
    # Display configuration
    click.echo(f"\n📍 API Endpoint: {Config.PARASAIL_BASE_URL}")
    click.echo(f"🤖 Vision Model: {Config.VISION_MODEL}")
    click.echo(f"📁 Documents Dir: {Config.DOCUMENTS_DIR}")
    click.echo(f"📤 Output Dir: {Config.OUTPUT_DIR}")
    click.echo(f"🔄 Max Retries: {Config.MAX_RETRIES}")
    click.echo(f"⏱️  Retry Delay: {Config.RETRY_DELAY}s")
    
    # Check directories
    click.echo(f"\n📂 Directory Status:")
    if Config.DOCUMENTS_DIR.exists():
        doc_count = len(list(Config.DOCUMENTS_DIR.glob('*')))
        click.echo(f"   ✅ Documents: {doc_count} files")
    else:
        click.echo(f"   ❌ Documents: Directory not found")
    
    if Config.OUTPUT_DIR.exists():
        out_count = sum(1 for _ in Config.OUTPUT_DIR.rglob('*') if _.is_file())
        click.echo(f"   ✅ Outputs: {out_count} files")
    else:
        click.echo(f"   ❌ Outputs: Directory not found")

# =====================================================
# MAIN ENTRY POINT
# =====================================================

if __name__ == "__main__":
    """
    Main entry point for the KYC processing system using Parasail.io.
    
    Ensures proper setup and provides helpful error messages if configuration is missing.
    """
    # Check for API key
    if not Config.PARASAIL_API_KEY:
        click.echo("⚠️  Warning: PARASAIL_API_KEY environment variable not set!")
        click.echo("Please ensure you have a .env file with: PARASAIL_API_KEY='your_actual_key'")
        click.echo(f"Looking for .env in: {Path.cwd().parent}")
    
    # Ensure documents directory exists
    if not Config.DOCUMENTS_DIR.exists():
        Config.DOCUMENTS_DIR.mkdir(exist_ok=True)
        click.echo(f"📁 Created documents directory: {Config.DOCUMENTS_DIR}")
        click.echo("Please add identity document images to this directory")
    
    # Run CLI
    cli()