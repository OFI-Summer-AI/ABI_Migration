import sys
from pathlib import Path

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


# ─── Terminal Formatting Helpers ──────────────────────────────────────────────

class TermStyle:
    """ANSI escape codes for styled terminal output."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    # Colors
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    BLUE    = "\033[94m"

S = TermStyle

BANNER = f"""
{S.CYAN}{S.BOLD}+--------------------------------------------------------------+
|               CELONIS  DATA  EXTRACTION  TOOL               |
+--------------------------------------------------------------+{S.RESET}
"""

def header(title: str) -> str:
    """Section header with ASCII characters."""
    width = 60
    line = "-" * width
    return (
        f"\n{S.CYAN}+{line}+{S.RESET}\n"
        f"{S.CYAN}|{S.BOLD}{S.WHITE}  {title.upper():<{width - 2}}{S.RESET}{S.CYAN}|{S.RESET}\n"
        f"{S.CYAN}+{line}+{S.RESET}"
    )

def step(icon: str, msg: str):
    """Print a step indicator."""
    print(f"  {icon}  {msg}")

def step_ok(msg: str):
    step(f"{S.GREEN}[OK]{S.RESET}", msg)

def step_info(msg: str):
    step(f"{S.CYAN}[INFO]{S.RESET}", msg)

def step_warn(msg: str):
    step(f"{S.YELLOW}[WARN]{S.RESET}", f"{S.YELLOW}{msg}{S.RESET}")

def step_fail(msg: str):
    step(f"{S.RED}[FAIL]{S.RESET}", f"{S.RED}{msg}{S.RESET}")

def kv(key: str, value, indent: int = 6):
    """Print a key-value pair with consistent alignment."""
    pad = " " * indent
    print(f"{pad}{S.DIM}{key:<20}{S.RESET} {value}")

def divider():
    print(f"  {S.DIM}{'-' * 56}{S.RESET}")

def result_box(title: str, rows: list[tuple[str, str]]):
    """Print a compact results box."""
    width = 56
    line = "-" * width
    print(f"\n  {S.GREEN}+{line}+{S.RESET}")
    print(f"  {S.GREEN}|{S.BOLD}  [OK] {title:<{width - 7}}{S.RESET}{S.GREEN}|{S.RESET}")
    print(f"  {S.GREEN}+{line}+{S.RESET}")
    for k, v in rows:
        content = f"  {k:<22} {v}"
        print(f"  {S.GREEN}|{S.RESET}{content:<{width}}{S.GREEN}|{S.RESET}")
    print(f"  {S.GREEN}+{line}+{S.RESET}")


# ─── Phase 1: Configuration & Connection ─────────────────────────────────────

def init_pipeline():
    """Load settings, create logger, connect to Celonis. Returns (settings, logger, connector)."""
    print(header("CONFIGURATION"))

    # Settings
    try:
        settings = get_settings()
        logger = get_logger(
            "extractor",
            log_dir=settings.log_dir,
            level=settings.log_level,
        )
        step_ok("Settings loaded from .env")
        kv("Celonis URL", settings.celonis_url)
        kv("Output dir", settings.output_dir)
    except Exception as e:
        step_fail(f"Configuration error: {e}")
        print(f"\n  Create a {S.BOLD}.env{S.RESET} file with:")
        print(f"    CELONIS_URL=https://your-team.celonis.cloud")
        print(f"    CELONIS_API_TOKEN=your-token\n")
        sys.exit(1)

    # Connection
    print(header("CONNECTION"))
    step_info("Connecting to Celonis …")
    try:
        connector = CelonisConnector(
            settings.celonis_url,
            settings.celonis_api_token,
        )
        if not connector.test_connection():
            raise ConnectionError("Connection test returned False")
        step_ok("Connected successfully")
    except Exception as e:
        step_fail(f"Connection failed: {e}")
        print(f"  Check your CELONIS_URL and CELONIS_API_TOKEN\n")
        sys.exit(1)

    return settings, logger, connector

def run_kpi_extraction(settings, logger, connector):
    """Extract KPIs from a Knowledge Model (or legacy View)."""
    print(header("KPI EXTRACTION"))

    extractor = CelonisExtractor(connector, space_id=settings.space_id, package_id=settings.package_id, quiet=True)
    file_handler = FileHandler(settings.output_dir)

    # Determine source
    if settings.knowledge_model_id:
        step_info(f"Source: Knowledge Model  ({S.DIM}{settings.knowledge_model_id}{S.RESET})")
    elif settings.analysis_id:
        step_info(f"Source: View / Analysis  ({S.DIM}{settings.analysis_id}{S.RESET})")
    else:
        step_fail("No KNOWLEDGE_MODEL_ID or ANALYSIS_ID configured")
        sys.exit(1)

    # Extract
    step_info("Extracting KPIs, Filters & Attributes …")
    try:
        if settings.knowledge_model_id:
            result = extractor.extract_from_knowledge_model(settings.knowledge_model_id)
        else:
            result = extractor.extract_from_view(settings.analysis_id)
    except Exception as e:
        step_fail(f"Extraction failed: {e}")
        raise

    # Save
    step_info("Saving results …")
    try:
        kpi_dicts = [kpi.model_dump() for kpi in result.kpis]
        json_path = file_handler.save_json({"kpis": kpi_dicts}, "extracted_kpis.json")
        stats = {
            "total": result.total_count,
            "by_type": result.by_type,
            "by_record": result.by_record,
            "duration": result.duration_seconds,
        }
        report_path = file_handler.generate_report(stats)
    except Exception as e:
        step_fail(f"Failed to save results: {e}")
        sys.exit(1)

    # Summary
    type_breakdown = ", ".join(f"{t}: {c}" for t, c in result.by_type.items()) or "—"
    result_box("KPI Extraction Complete", [
        ("Duration",      f"{result.duration_seconds:.2f}s"),
        ("Total items",   str(result.total_count)),
        ("Breakdown",     type_breakdown),
        ("JSON output",   str(json_path)),
        ("Report",        str(report_path)),
    ])
    return result


# ─── Phase 3: Transformation Extraction ──────────────────────────────────────

def run_transformation_extraction(settings, logger, connector):
    """Extract SQL transformations from a Data Pool."""
    print(header("TRANSFORMATION EXTRACTION"))

    extractor = CelonisExtractor(connector, space_id=settings.space_id, package_id=settings.package_id, quiet=True)
    file_handler = FileHandler(settings.output_dir)

    pool_identifier = settings.data_pool_name or settings.data_pool_id
    if not pool_identifier:
        step_fail("No DATA_POOL_NAME or DATA_POOL_ID configured in .env")
        sys.exit(1)

    step_info(f"Data Pool: {S.DIM}{pool_identifier}{S.RESET}")
    step_info("Extracting transformations (jobs → tasks) …")

    try:
        result = extractor.extract_transformations(pool_identifier)
    except Exception as e:
        step_fail(f"Extraction failed: {e}")
        raise

    # Save
    step_info("Saving results …")
    try:
        trans_dicts = [t.model_dump() for t in result.transformations]
        json_path = file_handler.save_json(trans_dicts, "extracted_transformations.json")

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
    except Exception as e:
        step_fail(f"Failed to save results: {e}")
        sys.exit(1)

    # Summary
    job_count = len(result.by_job)
    status_info = ", ".join(f"{s}: {c}" for s, c in result.by_status.items()) or "—"
    result_box("Transformation Extraction Complete", [
        ("Duration",          f"{result.duration_seconds:.2f}s"),
        ("Total transforms",  str(result.total_count)),
        ("Jobs scanned",      str(job_count)),
        ("Status breakdown",  status_info),
        ("JSON output",       str(json_path)),
        ("Report",            str(report_path)),
    ])
    return result


# ─── Phase 4: Data Model Extraction ──────────────────────────────────────────

def run_data_model_extraction(settings, logger, connector):
    """Extract metadata (and optionally data) from a Data Model."""
    print(header("DATA MODEL EXTRACTION"))

    extractor = CelonisExtractor(connector, space_id=settings.space_id, package_id=settings.package_id, quiet=True)
    file_handler = FileHandler(settings.output_dir)
    json_path = None
    report_path = None

    if not settings.data_model_id:
        step_fail("No DATA_MODEL_ID configured in .env")
        sys.exit(1)

    step_info(f"Data Model ID: {S.DIM}{settings.data_model_id}{S.RESET}")
    # Get tables to extract from settings
    table_names = None
    if settings.data_model_tables:
        table_names = [t.strip() for t in settings.data_model_tables.split(",") if t.strip()]
        step_info(f"Target tables found in settings: {S.BOLD}{', '.join(table_names)}{S.RESET}")
    else:
        step_info(f"Target tables not found in settings: {S.DIM}{settings.data_model_tables}{S.RESET}")

    try:
        result = extractor.extract_from_data_model(settings.data_model_id, table_names=table_names)
    except Exception as e:
        step_fail(f"Extraction failed: {e}")
        raise

    # Extraction of row data skipped (User request: metadata only)
    data_file_path = None

    # Save metadata
    step_info("Saving metadata …")
    try:
        table_dicts = [t.model_dump() for t in result.tables]
        metadata = {
            "data_model_id": result.data_model_id,
            "data_model_name": result.data_model_name,
            "table_count": result.table_count,
            "total_columns": result.total_columns,
            "total_rows": result.total_rows,
            "tables": table_dicts,
            "extraction_duration": result.duration_seconds,
            "errors": result.errors,
        }
        json_path = file_handler.save_json(metadata, "data_model_metadata.json")

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
    except Exception as e:
        step_fail(f"Failed to save results: {e}")
        sys.exit(1)

    # Summary
    rows = [
        ("Duration",      f"{result.duration_seconds:.2f}s"),
        ("Data Model",    result.data_model_name),
        ("Tables",        str(result.table_count)),
        ("Total columns", str(result.total_columns)),
        ("Total rows",    str(result.total_rows)),
        ("JSON output",   str(json_path)),
        ("Report",        str(report_path)),
    ]
    if data_file_path:
        rows.append(("Data export", str(data_file_path)))
    result_box("Data Model Extraction Complete", rows)

    if result.errors:
        print()
        for err in result.errors:
            step_warn(err)

    return result

def run_pipeline(kpis=True, transformations=True, data_model=False):
    """Orchestrate the full pipeline with clean output."""
    print(BANNER)

    settings, logger, connector = init_pipeline()

    phases_run = []

    if kpis:
        run_kpi_extraction(settings, logger, connector)
        phases_run.append("KPIs")

    if transformations:
        run_transformation_extraction(settings, logger, connector)
        phases_run.append("Transformations")

    if data_model:
        run_data_model_extraction(settings, logger, connector)
        phases_run.append("Data Model")

    # Footer
    print(f"\n{S.CYAN}{S.BOLD}{'─' * 62}{S.RESET}")
    print(f"  {S.GREEN}{S.BOLD}ALL DONE{S.RESET}  –  Extracted: {', '.join(phases_run)}")
    print(f"{S.CYAN}{S.BOLD}{'─' * 62}{S.RESET}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "--transformations":
            run_pipeline(kpis=False, transformations=True, data_model=False)
        elif command == "--kpis-only":
            run_pipeline(kpis=True, transformations=False, data_model=False)
        elif command == "--data-model":
            run_pipeline(kpis=False, transformations=False, data_model=True)
        elif command == "--all":
            run_pipeline(kpis=True, transformations=True, data_model=True)
        else:
            print(f"  {S.RED}Unknown command:{S.RESET} {command}\n")
            print(f"  {S.BOLD}Usage:{S.RESET}")
            print(f"    python main_ingest.py                   {S.DIM}# KPIs + Transformations{S.RESET}")
            print(f"    python main_ingest.py --kpis-only        {S.DIM}# KPIs only{S.RESET}")
            print(f"    python main_ingest.py --transformations   {S.DIM}# Transformations only{S.RESET}")
            print(f"    python main_ingest.py --data-model        {S.DIM}# Data Model only{S.RESET}")
            print(f"    python main_ingest.py --all               {S.DIM}# Everything{S.RESET}")
            sys.exit(1)
    else:
        # Default: KPIs + Transformations
        run_pipeline(kpis=True, transformations=True, data_model=False)