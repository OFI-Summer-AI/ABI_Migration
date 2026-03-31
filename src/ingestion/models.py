"""
Data models for using Pydantic for validation.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class CelonisKPI(BaseModel):
    """
    Represents a single KPI or Attribute extracted from Celonis.
    """
    kpi_id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Display name")
    pql_formula: str = Field(..., description="PQL formula/query")
    record_id: str = Field(..., description="Parent record/component ID")
    attribute_type: str = Field(..., description="KPI, Attribute, or Component")
    raw_metadata: Dict[str, Any] = Field(default_factory=dict, description="Original object data")
    extracted_at: datetime = Field(default_factory=datetime.now, description="Extraction timestamp")
    depends_on: List[str] = Field(default_factory=list, description="List of KPI IDs this formula depends on")

# Config class for JSON encoders.
class Config:
    json_encoders = {
        datetime: lambda v: v.isoformat()
    }

# ExtractionResult class to store extraction results.
class ExtractionResult(BaseModel):
    """
    Container for extraction results with statistics.
    """
    kpis: List[CelonisKPI] = Field(default_factory=list)
    total_count: int = Field(default=0)
    by_type: Dict[str, int] = Field(default_factory=dict)
    by_record: Dict[str, int] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # Add KPI and update statistics.
    def add_kpi(self, kpi: CelonisKPI):
        """Add KPI and update statistics."""
        self.kpis.append(kpi)
        self.total_count += 1
        
        # Update type count
        self.by_type[kpi.attribute_type] = self.by_type.get(kpi.attribute_type, 0) + 1
        
        # Update record count
        self.by_record[kpi.record_id] = self.by_record.get(kpi.record_id, 0) + 1
    
    # Mark extraction as complete.
    def finalize(self):
        """Mark extraction as complete."""
        self.completed_at = datetime.now()
    
    @property
    def duration_seconds(self) -> float:
        """Calculate extraction duration."""
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

# CelonisTransformation class to store transformation results.
class CelonisTransformation(BaseModel):
    """
    Represents a single Transformation extracted from Celonis Data Pool.
    """
    job_name: str = Field(..., description="Parent job name")
    transformation_name: str = Field(..., description="Transformation/task name")
    sap_tables_used: str = Field(default="N/A", description="Comma-separated SAP tables")
    status: str = Field(default="Success", description="Extraction status")
    raw_sql: str = Field(default="", description="SQL transformation code")
    extracted_at: datetime = Field(default_factory=datetime.now, description="Extraction timestamp")

# Config class for JSON encoders.
class Config:
    json_encoders = {
        datetime: lambda v: v.isoformat()
    }

# TransformationExtractionResult class to store transformation extraction results.
class TransformationExtractionResult(BaseModel):
    """
    Container for transformation extraction results with statistics.
    """
    
    transformations: List[CelonisTransformation] = Field(default_factory=list)
    total_count: int = Field(default=0)
    by_job: Dict[str, int] = Field(default_factory=dict)
    by_status: Dict[str, int] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def add_transformation(self, transformation: CelonisTransformation):
        """Add transformation and update statistics."""
        self.transformations.append(transformation)
        self.total_count += 1
        
        # Update job count
        self.by_job[transformation.job_name] = self.by_job.get(transformation.job_name, 0) + 1
        
        # Update status count
        self.by_status[transformation.status] = self.by_status.get(transformation.status, 0) + 1
    
    def finalize(self):
        """Mark extraction as complete."""
        self.completed_at = datetime.now()
    
    @property
    def duration_seconds(self) -> float:
        """Calculate extraction duration."""
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

# DataModelTableInfo class to store metadata about a table in a data model.
class DataModelTableInfo(BaseModel):
    """
    Represents metadata about a table in a data model.
    """
    table_name: str = Field(..., description="Name of the table")
    column_count: int = Field(default=0, description="Number of columns")
    row_count: Optional[int] = Field(default=None, description="Number of rows (if available)")
    columns: List[str] = Field(default_factory=list, description="List of column names")
    description: Optional[str] = Field(default="", description="Table description")
    extracted_at: datetime = Field(default_factory=datetime.now, description="Extraction timestamp")

# Config class for JSON encoders.
class Config:
    json_encoders = {
        datetime: lambda v: v.isoformat()
    }

# DataModelExtractionResult class to store data model extraction results.
class DataModelExtractionResult(BaseModel):
    """
    Container for data model extraction results.
    """
    data_model_id: str = Field(..., description="Data model ID")
    data_model_name: str = Field(..., description="Data model name")
    tables: List[DataModelTableInfo] = Field(default_factory=list, description="List of tables")
    table_count: int = Field(default=0, description="Total table count")
    total_rows: int = Field(default=0, description="Total rows extracted")
    total_columns: int = Field(default=0, description="Total columns")
    errors: List[str] = Field(default_factory=list, description="Any extraction errors")
    started_at: datetime = Field(default_factory=datetime.now, description="Extraction start time")
    completed_at: Optional[datetime] = None
    data_file_path: Optional[str] = Field(default=None, description="Path to saved data file")

    # Add table info and update statistics.
    def add_table(self, table_info: DataModelTableInfo):
        """Add table info and update statistics."""
        self.tables.append(table_info)
        self.table_count += 1
        self.total_columns += table_info.column_count
        if table_info.row_count:
            self.total_rows += table_info.row_count

    # Mark extraction as complete.
    def finalize(self):
        """Mark extraction as complete."""
        self.completed_at = datetime.now()

    @property
    def duration_seconds(self) -> float:
        """Calculate extraction duration."""
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()
