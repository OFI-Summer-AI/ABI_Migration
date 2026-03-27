import sys
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from src.ingestion.extractor import CelonisExtractor
from src.ingestion.connectors import CelonisConnector
from src.ingestion.models import ExtractionResult, DataModelExtractionResult
from src.utils.logger import get_logger
from src.utils.file_handler import FileHandler


def print_results(result: ExtractionResult, output_files: dict):
    """Print extraction summary."""
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Duration:      {result.duration_seconds:.2f} seconds")
    print(f"Total KPIs:    {result.total_count}")
    print(f"By Type:       {result.by_type}")
    print("-" * 60)
    print("Output Files:")
    for name, path in output_files.items():
        print(f"  {name}: {path}")
    print("=" * 60)


def print_data_model_results(result: DataModelExtractionResult, output_files: dict):
    """Print data model extraction summary."""
    print("\n" + "=" * 60)
    print("DATA MODEL EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Data Model:    {result.data_model_name} ({result.data_model_id})")
    print(f"Duration:      {result.duration_seconds:.2f} seconds")
    print(f"Total Tables:  {result.table_count}")
    print(f"Total Columns: {result.total_columns}")
    print(f"Total Rows:    {result.total_rows}")
    if result.tables:
        print("\nTables extracted:")
        for table in result.tables:
            print(f"  - {table.table_name}: {table.column_count} columns, {table.row_count} rows")
    print("-" * 60)
    print("Output Files:")
    for name, path in output_files.items():
        print(f"  {name}: {path}")
    if result.errors:
        print("-" * 60)
        print("Warnings/Errors:")
        for error in result.errors:
            print(f"  ⚠ {error}")
    print("=" * 60)


def main():
    """Main execution function."""
    
    # Load configuration
    try:
        settings = get_settings()
        logger = get_logger(
            "extractor",
            log_dir=settings.log_dir,
            level=settings.log_level
        )
        logger.info("Configuration loaded successfully")
    except Exception as e:
        print(f"Configuration error: {e}")
        print("\nCreate a .env file with:")
        print("CELONIS_URL=https://your-team.celonis.cloud")
        print("CELONIS_API_TOKEN=your-token")
        sys.exit(1)
    
    # Initialize components
    try:
        connector = CelonisConnector(
            settings.celonis_url,
            settings.celonis_api_token
        )
        
        # Test connection
        logger.info("Testing Celonis connection...")
        if not connector.test_connection():
            raise ConnectionError("Failed to connect to Celonis")
        logger.info("Connection successful")
        
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        print(f"\n Cannot connect to Celonis: {e}")
        print("Check your CELONIS_URL and CELONIS_API_TOKEN")
        sys.exit(1)
    
   
    # Run extraction
    extractor = CelonisExtractor(connector, space_id=settings.space_id, package_id=settings.package_id)
    file_handler = FileHandler(settings.output_dir)
    
    try:
        # Try Knowledge Model first (has actual PQL formulas)
        if settings.knowledge_model_id:
            logger.info(f"Extracting from Knowledge Model: {settings.knowledge_model_id}")
            result = extractor.extract_from_knowledge_model(settings.knowledge_model_id)
        elif settings.analysis_id:
            logger.info(f"Extracting from View: {settings.analysis_id}")
            result = extractor.extract_from_view(settings.analysis_id)
        else:
            raise ValueError("No KNOWLEDGE_MODEL_ID or ANALYSIS_ID configured")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise

    # Save results
    try:
        # Convert to dicts for serialization
        kpi_dicts = [kpi.model_dump() for kpi in result.kpis]
        
        # Save JSON
        wrapped_json = {"kpis": kpi_dicts}
        json_path = file_handler.save_json(wrapped_json, "extracted_kpis.json")
        logger.info(f"Saved JSON: {json_path}")
        
        # Generate report
        stats = {
            "total": result.total_count,
            "by_type": result.by_type,
            "by_record": result.by_record,
            "duration": result.duration_seconds
        }
        report_path = file_handler.generate_report(stats)
        
        output_files = {
            "JSON": json_path,
            "Report": report_path
        }
        
        print_results(result, output_files)
        logger.info("Extraction completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        print(f"\n Error saving results: {e}")
        sys.exit(1)


def extract_transformations():
    """Extract transformations from data pool."""
    # Load configuration
    try:
        settings = get_settings()
        logger = get_logger(
            "transformations",
            log_dir=settings.log_dir,
            level=settings.log_level
        )
        logger.info("Configuration loaded successfully")
    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    # Initialize components
    try:
        connector = CelonisConnector(
            settings.celonis_url,
            settings.celonis_api_token
        )
        
        logger.info("Testing Celonis connection...")
        if not connector.test_connection():
            raise ConnectionError("Failed to connect to Celonis")
        logger.info("Connection successful")
        
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        print(f"\n Cannot connect to Celonis: {e}")
        sys.exit(1)
    
    # Run extraction
    extractor = CelonisExtractor(connector, space_id=settings.space_id, package_id=settings.package_id)
    file_handler = FileHandler(settings.output_dir)
    
    try:
        pool_identifier = settings.data_pool_name or settings.data_pool_id
        if not pool_identifier:
            raise ValueError("No DATA_POOL_NAME or DATA_POOL_ID configured in .env")
        
        logger.info(f"Extracting transformations from: {pool_identifier}")
        result = extractor.extract_transformations(pool_identifier)
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise
    
    # Save results
    try:
        # Convert to dicts
        trans_dicts = [t.model_dump() for t in result.transformations]
        
        # Save JSON
        json_path = file_handler.save_json(trans_dicts, "extracted_transformations.json")
        logger.info(f"Saved JSON: {json_path}")
        
        # Generate report
        report_path = Path(settings.output_dir) / f"transformation_report_{file_handler.timestamp}.txt"
        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("CELONIS TRANSFORMATION EXTRACTION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {file_handler.timestamp}\n")
            f.write(f"Total Transformations: {result.total_count}\n")
            f.write(f"By Job: {result.by_job}\n")
            f.write(f"By Status: {result.by_status}\n")
            f.write("=" * 60 + "\n")
        
        output_files = {
            "JSON": json_path,
            "Report": report_path
        }
        
        print("\n" + "=" * 60)
        print("TRANSFORMATION EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Duration:      {result.duration_seconds:.2f} seconds")
        print(f"Total Trans:   {result.total_count}")
        print(f"By Job:        {result.by_job}")
        print(f"By Status:     {result.by_status}")
        print("-" * 60)
        print("Output Files:")
        for name, path in output_files.items():
            print(f"  {name}: {path}")
        print("=" * 60)
        
        logger.info("Transformation extraction completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        print(f"\n Error saving results: {e}")
        sys.exit(1)


def extract_data_model():
    """Extract metadata and optionally data from a data model."""
    # Load configuration
    try:
        settings = get_settings()
        logger = get_logger(
            "data_model_extractor",
            log_dir=settings.log_dir,
            level=settings.log_level
        )
        logger.info("Configuration loaded successfully")
    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    # Initialize components
    try:
        connector = CelonisConnector(
            settings.celonis_url,
            settings.celonis_api_token
        )
        
        logger.info("Testing Celonis connection...")
        if not connector.test_connection():
            raise ConnectionError("Failed to connect to Celonis")
        logger.info("Connection successful")
        
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        print(f"\n Cannot connect to Celonis: {e}")
        sys.exit(1)
    
    # Run extraction
    extractor = CelonisExtractor(connector, space_id=settings.space_id, package_id=settings.package_id)
    file_handler = FileHandler(settings.output_dir)
    
    try:
        if not settings.data_model_id:
            raise ValueError("No DATA_MODEL_ID configured in .env")
        
        logger.info(f"Extracting Data Model: {settings.data_model_id}")
        
        # Extract metadata
        result = extractor.extract_from_data_model(settings.data_model_id)
        
        # Optionally extract actual data
        data_file_path = None
        if getattr(settings, 'extract_data', False):
            try:
                logger.info("Extracting data from model...")
                use_chunking = getattr(settings, 'use_chunking', False)
                chunksize = getattr(settings, 'chunksize', 10000)
                
                df = extractor.extract_data_from_model(
                    settings.data_model_id,
                    use_chunking=use_chunking,
                    chunksize=chunksize
                )
                
                # Save data to file
                data_file_path = file_handler.save_dataframe(df, "data_model_export.csv")
                result.data_file_path = str(data_file_path)
                logger.info(f"Data exported to: {data_file_path}")
            except Exception as e:
                logger.warning(f"Could not extract data: {e}")
                result.errors.append(f"Data extraction: {e}")
        
    except Exception as e:
        logger.error(f"Data Model extraction failed: {e}")
        raise
    
    # Save results
    try:
        # Save metadata as JSON
        table_dicts = [t.model_dump() for t in result.tables]
        metadata = {
            "data_model_id": result.data_model_id,
            "data_model_name": result.data_model_name,
            "table_count": result.table_count,
            "total_columns": result.total_columns,
            "total_rows": result.total_rows,
            "tables": table_dicts,
            "extraction_duration": result.duration_seconds,
            "errors": result.errors
        }
        
        json_path = file_handler.save_json(metadata, "data_model_metadata.json")
        logger.info(f"Saved metadata: {json_path}")
        
        # Generate report
        report_path = Path(settings.output_dir) / f"data_model_report_{file_handler.timestamp}.txt"
        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("CELONIS DATA MODEL EXTRACTION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Data Model: {result.data_model_name}\n")
            f.write(f"Model ID: {result.data_model_id}\n")
            f.write(f"Timestamp: {file_handler.timestamp}\n")
            f.write(f"Duration: {result.duration_seconds:.2f} seconds\n")
            f.write(f"Total Tables: {result.table_count}\n")
            f.write(f"Total Columns: {result.total_columns}\n")
            f.write(f"Total Rows: {result.total_rows}\n")
            f.write("=" * 60 + "\n")
            
            if result.tables:
                f.write("\nTABLE DETAILS:\n")
                f.write("-" * 60 + "\n")
                for table in result.tables:
                    f.write(f"\nTable: {table.table_name}\n")
                    f.write(f"  Columns: {table.column_count}\n")
                    f.write(f"  Rows: {table.row_count}\n")
                    if table.columns:
                        f.write(f"  Fields: {', '.join(table.columns[:10])}")
                        if len(table.columns) > 10:
                            f.write(f" + {len(table.columns) - 10} more")
                        f.write("\n")
            
            if result.errors:
                f.write("\n" + "=" * 60 + "\n")
                f.write("WARNINGS/ERRORS:\n")
                f.write("-" * 60 + "\n")
                for error in result.errors:
                    f.write(f"⚠ {error}\n")
        
        output_files = {
            "Metadata": json_path,
            "Report": report_path
        }
        
        if data_file_path:
            output_files["Data"] = str(data_file_path)
        
        print_data_model_results(result, output_files)
        logger.info("Data Model extraction completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        print(f"\n Error saving results: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "--transformations":
            extract_transformations()
        elif command == "--kpis-only":
            main()
        elif command == "--data-model":
            extract_data_model()
        elif command == "--all":
            # Extract all: KPIs, Transformations, and Data Model
            print("\n" + "="*60)
            print("EXTRACTING KPIs, TRANSFORMATIONS, AND DATA MODEL")
            print("="*60)
            main()
            print("\n")
            extract_transformations()
            print("\n")
            extract_data_model()
        else:
            print(f"Unknown command: {command}")
            print("\nUsage:")
            print("  python main.py [--kpis-only|--transformations|--data-model|--all]")
            sys.exit(1)
    else:
        # Default: Extract both KPIs and Transformations
        print("\n" + "="*60)
        print("EXTRACTING KPIs AND TRANSFORMATIONS")
        print("="*60)
        main()
        print("\n")
        extract_transformations()