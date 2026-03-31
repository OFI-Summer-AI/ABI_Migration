import os
import re
import sys
import io
import pandas as pd
from typing import Any, Optional
from tqdm import tqdm
from .models import (CelonisKPI, ExtractionResult, CelonisTransformation, TransformationExtractionResult,DataModelTableInfo, DataModelExtractionResult)
from .connectors import CelonisConnector
from ..utils.logger import get_logger

# Extractor class to handle KPI, transformation, and data model extraction logic.
class CelonisExtractor:
    # Initialize with a CelonisConnector and identifiers for space and package.
    def __init__(self, connector: CelonisConnector, space_id: str, package_id: str, quiet: bool = False):
        if not space_id or not package_id:
            raise ValueError("space_id and package_id are required")
        self.connector = connector
        self.space_id = space_id
        self.package_id = package_id
        self.logger = get_logger(__name__)
        # When quiet=True, suppress tqdm progress bars (caller provides its own UI)
        self._tqdm_file = open(os.devnull, "w") if quiet else sys.stderr
        self._quiet = quiet

    # Extract KPIs, Filters, and Attributes from a Knowledge Model via get_content().
    def extract_from_knowledge_model(self, km_id: str) -> ExtractionResult:
        self.logger.info(f"Extracting KM: {km_id}")
        result = ExtractionResult()

        try:
            km = self.connector.get_knowledge_model(self.space_id, self.package_id, km_id)
            self.logger.info(f"Connected to KM: {km.name}")

            content = km.get_content()
            if content is None:
                self.logger.warning("KM content is None — no KPIs, Filters, or Attributes will be extracted.")
                result.finalize()
                return result

            # 1. Top-level KPIs
            kpi_list = list(content.kpis or [])
            self.logger.info(f"Found {len(kpi_list)} top-level KPIs in content")
            for kpi in tqdm(kpi_list, desc="KPIs", file=self._tqdm_file, disable=self._quiet):
                if kpi is None:
                    continue
                obj = self._build_kpi_from_metadata(kpi, "KPI", record_id="knowledge_model")
                if obj:
                    result.add_kpi(obj)

            # 2. Top-level Filters
            filter_list = list(content.filters or [])
            self.logger.info(f"Found {len(filter_list)} top-level Filters in content")
            for f in tqdm(filter_list, desc="Filters", file=self._tqdm_file, disable=self._quiet):
                if f is None:
                    continue
                obj = self._build_kpi_from_metadata(f, "Filter", record_id="knowledge_model")
                if obj:
                    result.add_kpi(obj)

            # 3. Attributes inside Records (most PQL formulas live here)
            record_list = list(content.records or [])
            self.logger.info(f"Found {len(record_list)} Records — extracting Attributes")
            for record in tqdm(record_list, desc="Records", file=self._tqdm_file, disable=self._quiet):
                if record is None:
                    continue
                record_id = getattr(record, "id", "unknown_record")
                for attr in (record.attributes or []):
                    if attr is None:
                        continue
                    obj = self._build_kpi_from_metadata(attr, "Attribute", record_id=record_id)
                    if obj:
                        result.add_kpi(obj)

            result.finalize()
            self.logger.info(
                f"KM extraction complete — {result.total_count} items "
                f"(KPIs={result.by_type.get('KPI', 0)}, "
                f"Filters={result.by_type.get('Filter', 0)}, "
                f"Attributes={result.by_type.get('Attribute', 0)})"
            )

        except Exception as e:
            self.logger.error(f"KM extraction failed: {e}")
            result.errors.append(str(e))
            raise

        return result

    # Extract transformations from a data pool.
    def extract_transformations(self, pool_name: str) -> TransformationExtractionResult:
        self.logger.info(f"Extracting transformations from: {pool_name}")
        result = TransformationExtractionResult()

        try:
            pool = self.connector.get_data_pool(pool_name)

            for job in tqdm(pool.get_jobs(), desc="Jobs", file=self._tqdm_file, disable=self._quiet):
                for task in job.get_tasks():
                    result.add_transformation(self._build_transformation(job, task))

            result.finalize()

        except Exception as e:
            self.logger.error(f"Transformation extraction failed: {e}")
            result.errors.append(str(e))
            raise

        return result

    # Extract data model metadata and data from a Data Model.
    def extract_from_data_model(self, data_model_id: str, table_names: Optional[list[str]] = None, use_chunking: bool = False, chunksize: int = 10000) -> DataModelExtractionResult:
        """
        Extract metadata and data from a Celonis Data Model.
        Args:
            data_model_id: ID of the data model to extract from
            table_names: Optional list of table names to filter metadata extraction
            use_chunking: Whether to use chunking for large datasets (>1GB)
            chunksize: Number of rows per chunk if chunking is enabled
            
        Returns:
            DataModelExtractionResult containing metadata and extracted data
        """
        self.logger.info(f"Extracting Data Model: {data_model_id}")
        
        # Initialize result early to avoid UnboundLocalError in 'except'
        result = DataModelExtractionResult(
            data_model_id=data_model_id,
            data_model_name="Unknown"
        )
        
        try:
            # Get data model
            datamodel = self.connector.get_data_model(data_model_id)
            self.logger.info(f"Connected to Data Model: {datamodel.name}")
            result.data_model_name = datamodel.name
            
            # Extract table metadata
            self.logger.info("Extracting table metadata...")
            self._extract_table_metadata(datamodel, result, table_names=table_names)
            
            result.finalize()
            self.logger.info(f"Data Model extraction complete. Tables: {result.table_count}")
            
        except Exception as e:
            self.logger.error(f"Data Model extraction failed: {e}")
            result.errors.append(str(e))
            raise
        
        return result
    
    # Extract actual data from a Celonis Data Model using PQL.
    def extract_data_from_model(self, data_model_id: str, use_chunking: bool = False, chunksize: int = 10000) -> pd.DataFrame:
        """
        Extract actual data from a Celonis Data Model using PQL.
        Args:
            data_model_id: ID of the data model
            use_chunking: Whether to use chunking for large datasets
            chunksize: Rows per chunk if chunking enabled
            
        Returns:
            DataFrame with extracted data
        """
        self.logger.info(f"Extracting data from Data Model: {data_model_id}")
        
        try:
            # Import PQL here to avoid circular imports
            from pycelonis import pql
            
            # Get data model
            datamodel = self.connector.get_data_model(data_model_id)
            self.logger.info(f"Connected to Data Model: {datamodel.name}")
            
            # Create empty PQL query (will get all columns)
            q = pql.PQL()
            
            # Extract data with optional chunking
            if use_chunking:
                self.logger.info(f"Extracting data with chunking (chunksize={chunksize})...")
                df = datamodel.get_data_frame(q, chunksize=chunksize)
            else:
                self.logger.info("Extracting data without chunking...")
                df = datamodel.get_data_frame(q)
            
            self.logger.info(f"Data extraction complete. Shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Data extraction failed: {e}")
            raise
    
    # Extract data from a Data Model using a custom PQL query.
    def extract_data_with_pql(self, data_model_id: str, pql_query: Any, use_chunking: bool = False, chunksize: int = 10000) -> pd.DataFrame:
        """
        Extract data from a Celonis Data Model using a custom PQL query.
        Args:
            data_model_id: ID of the data model
            pql_query: PQL query object (from pycelonis.pql)
            use_chunking: Whether to use chunking for large datasets
            chunksize: Rows per chunk if chunking enabled
            
        Returns:
            DataFrame with extracted data
        """
        self.logger.info(f"Extracting data from Data Model: {data_model_id} with custom PQL")
        
        try:
            # Get data model
            datamodel = self.connector.get_data_model(data_model_id)
            self.logger.info(f"Connected to Data Model: {datamodel.name}")
            
            # Extract data with optional chunking
            if use_chunking:
                self.logger.info(f"Extracting data with chunking (chunksize={chunksize})...")
                df = datamodel.get_data_frame(pql_query, chunksize=chunksize)
            else:
                self.logger.info("Extracting data without chunking...")
                df = datamodel.get_data_frame(pql_query)
            
            self.logger.info(f"Data extraction complete. Shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Data extraction with PQL failed: {e}")
            raise

    # Extract all data from a specific table in a Data Model.
    def extract_table_data(self, data_model_id: str, table_name: str, use_chunking: bool = False, chunksize: int = 10000) -> pd.DataFrame:
        """
        Extract all data from a specific table in a Data Model.
        
        Args:
            data_model_id: ID of the data model
            table_name: Name of the table to extract
            use_chunking: Whether to use chunking
            chunksize: Rows per chunk
            
        Returns:
            DataFrame containing table data
        """
        self.logger.info(f"Extracting table data: {table_name} from Data Model: {data_model_id}")
        
        try:
            # Import PQL components
            from pycelonis.pql import PQL, PQLColumn
            
            # Get data model
            datamodel = self.connector.get_data_model(data_model_id)
            
            # Get table object
            tables = self._safe_get_tables(datamodel)
            table_obj = next((t for t in tables if getattr(t, "name", getattr(t, "id", "")) == table_name), None)
            
            if not table_obj:
                raise ValueError(f"Table '{table_name}' not found in Data Model")
            
            # Build query for all columns
            columns = self._safe_get_columns(table_obj)
            if not columns:
                 # Fallback if we can't get column metadata
                 self.logger.warning(f"Could not retrieve column metadata for {table_name}, attempting generic query.")
                 q = PQL() # This might or might not work depending on DM config
            else:
                q = PQL()
                for col in columns:
                    # Celonis PQL format: "Table"."Column"
                    pql_expr = f'"{table_name}"."{col}"'
                    q.add(PQLColumn(query=pql_expr, name=col))
            
            # Execute
            if use_chunking:
                df = datamodel.get_data_frame(q, chunksize=chunksize)
            else:
                df = datamodel.get_data_frame(q)
                
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to extract table data for {table_name}: {e}")
            raise

    # Helper to build CelonisKPI from any KpiMetadata / FilterMetadata / AttributeMetadata object.
    def _build_kpi_from_metadata(self, obj: Any, obj_type: str, record_id: str = "knowledge_model") -> Optional[CelonisKPI]:
        pql = getattr(obj, "pql", None)
        if not pql:
            return None

        kpi_id = str(getattr(obj, "id", ""))
        display_name = getattr(obj, "display_name", None) or getattr(obj, "name", None) or kpi_id

        pql_str = str(pql).strip()
        deps = re.findall(r'KPI\([\'"]([^\'"]+)[\'"]\)', pql_str)

        return CelonisKPI(
            kpi_id=kpi_id,
            name=str(display_name),
            pql_formula=pql_str,
            record_id=record_id,
            attribute_type=obj_type,
            raw_metadata={},
            depends_on=deps
        )

    # Legacy helper kept for backward compatibility (used nowhere now, safe to remove later).
    def _build_kpi(self, obj: Any, obj_type: str) -> Optional[CelonisKPI]:
        return self._build_kpi_from_metadata(obj, obj_type)
    
    # Helper to build Transformation objects from Celonis job/task entities.
    def _build_transformation(self, job: Any, task: Any) -> CelonisTransformation:
        sql = self._safe_get_sql(task)
        tables = self._extract_tables(sql)

        return CelonisTransformation(
            job_name=job.name,
            transformation_name=task.name,
            sap_tables_used=tables,
            status="Success" if sql else "Empty",
            raw_sql=sql
        )

    # Safely attempt to get SQL from a task, handling different possible attributes and errors.
    def _safe_get_sql(self, task: Any) -> str:
        for attr in ["get_statement", "statement", "query"]:
            try:
                value = getattr(task, attr, None)
                if callable(value):
                    res = value()
                    if res is not None:
                        return str(res)
                elif value is not None:
                    return str(value)
            except Exception:
                return "[[ HIDDEN OR ERROR ]]"
        return ""

    # Extract table names from SQL using regex, with handling for hidden/obfuscated SQL.
    def _extract_tables(self, sql: str) -> str:
        if not sql or "[[ HIDDEN" in sql:
            return "N/A"

        matches = re.findall(r'(?i)(?:FROM|JOIN)\s+\w+\.?(\w+)', sql)
        return ", ".join(sorted(set(m.upper() for m in matches))) or "N/A"
    
    # Helper to extract table metadata from a data model.
    def _extract_table_metadata(self, datamodel: Any, result: DataModelExtractionResult, table_names: Optional[list[str]] = None):
        """
        Extract metadata about tables in the data model.
        
        Args:
            datamodel: Celonis data model object
            result: DataModelExtractionResult to populate
            table_names: Optional list of table names to filter for
        """
        try:
        # Try to get tables from data model
            tables = self._safe_get_tables(datamodel)
            
            # Filter tables if names provided
            if table_names:
                tables = [t for t in tables if getattr(t, "name", getattr(t, "id", "")) in table_names]
            
            for table in tqdm(tables, desc="Tables", file=self._tqdm_file, disable=self._quiet):
                table_info = self._build_table_info(table)
                if table_info:
                    result.add_table(table_info)
                    self.logger.info(f"Extracted table: {table_info.table_name} ({table_info.column_count} columns)")
        
        except Exception as e:
            self.logger.warning(f"Could not extract detailed table metadata: {e}")
            result.errors.append(f"Table metadata extraction: {e}")
    
    # Helper to safely get tables from a data model.
    def _safe_get_tables(self, datamodel: Any) -> list:
        """
        Safely attempt to get tables from a data model.
        Tries multiple possible attributes.
        """
        for attr in ["get_tables", "tables", "get_dataframes"]:
            try:
                value = getattr(datamodel, attr, None)
                if callable(value):
                    tables = value()
                    if tables:
                        self.logger.info(f"Retrieved tables using '{attr}'")
                        return tables if isinstance(tables, list) else [tables]
                elif value:
                    return value if isinstance(value, list) else [value]
            except Exception as e:
                self.logger.debug(f"Attribute '{attr}' failed: {e}")
                continue
        
        self.logger.warning("Could not retrieve tables from data model")
        return []
    
    # Helper to build DataModelTableInfo from a table object.
    def _build_table_info(self, table: Any) -> Optional[DataModelTableInfo]:
        """
        Build DataModelTableInfo from a table object.
        """
        try:
            table_name = getattr(table, "name", getattr(table, "id", "Unknown"))
            
            # Try to get columns
            columns = self._safe_get_columns(table)
            column_count = len(columns)
            
            # Try to get row count
            row_count = self._safe_get_row_count(table)
            
            return DataModelTableInfo(
                table_name=str(table_name),
                column_count=column_count,
                row_count=row_count,
                columns=columns,
                description=str(getattr(table, "description", ""))
            )
        except Exception as e:
            self.logger.warning(f"Failed to build table info: {e}")
            return None
    
    # Helper to safely get column names from a table.
    def _safe_get_columns(self, table: Any) -> list:
        """
        Safely attempt to get column names from a table.
        """
        for attr in ["get_columns", "columns", "get_fields", "fields"]:
            try:
                value = getattr(table, attr, None)
                if callable(value):
                    cols = value()
                    if cols:
                        return [str(c.name if hasattr(c, 'name') else c) for c in cols]
                elif value:
                    return [str(c.name if hasattr(c, 'name') else c) for c in value]
            except Exception:
                continue
        return []
    
    # Helper to safely get row count from a table.
    def _safe_get_row_count(self, table: Any) -> Optional[int]:
        """
        Safely attempt to get row count from a table.
        """
        for attr in ["row_count", "get_row_count", "count"]:
            try:
                value = getattr(table, attr, None)
                if callable(value):
                    count = value()
                    if count is not None:
                        return int(count)
                elif value is not None:
                    return int(value)
            except Exception:
                continue
        return None