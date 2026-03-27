# Ingestion_Layer
# connector.py -->

- pycelonis.get_celonis --- 

- Returns a Celonis client object that gives access to:
. Studio
. Data Integration
. Knowledge Models
. Views, Spaces, Packages

# extractor.py -->
- This module defines a class CelonisExtractor that is responsible for extracting:

. KPIs and Filters from a Celonis Knowledge Model
. Transformations (SQL tasks) from a Data Pool

. It uses a connector (CelonisConnector) to interact with Celonis APIs and converts raw data into structured objects.

- Class: CelonisExtractor:
Acts as the main service layer to:

. Connect to Celonis
. Extract data (KPIs, filters, transformations)
. Convert it into structured Python objects

# Recent Enhancements (Summary)

- Added a robust SQL post-processing utility (`extract_raw_sql_meaning.py`) that reads `extracted_transformations.json`, extracts structured meaning from `raw_sql`, and saves outputs into a separate processed folder.
- Generated transformation artifacts in `data/processed/transformation_meaning/`:
. `raw_sql/` one SQL file per transformation
. `summaries/` one readable text interpretation per transformation
. `transformations_meaning.json` consolidated machine-readable summary
- Improved English-to-SQL system prompt (`src/translation/sql_translation.py`) to be generalized and rules-driven for PQL-to-SQL translation, not limited to KPI examples.
- Added KPI dependency graph builder (`kpi_dependency_graph.py`) that:
. Detects KPI references via `KPI("...")`
. Builds directed dependency relationships
. Detects cycles and computes dependency paths
. Exports graph outputs in JSON and DOT format under `data/processed/kpi_dependency_graph/`
- Refined dependency logic to highlight direct nested KPI format (`KPI("...")`) and output clear nested dependency chains.
- Updated semantic explanation flow (`src/translation/semantic_translation.py`) so nested KPI explanations use dependency context and expanded formulas for better plain-English interpretation.

# End Goals Achieved

- Clear extraction and interpretation pipeline for transformation SQL.
- Reusable and safer PQL-to-SQL translation guidance with strict output constraints.
- Dependency-aware KPI understanding for nested KPI formulas.
- Traceable outputs in both human-readable and machine-readable formats for downstream migration/review tasks.
