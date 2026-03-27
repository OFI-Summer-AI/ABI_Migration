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
