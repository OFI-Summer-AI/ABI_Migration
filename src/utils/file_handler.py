"""
File handling utilities for JSON and CSV export.
"""
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import pandas as pd


class FileHandler:
    """
    Handles saving and loading of extraction data.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_json(self, data: List[Dict[str, Any]], filename: str = None) -> Path:
        """
        Save data to JSON file with timestamp.
        
        Args:
            data: List of dictionaries to save
            filename: Optional custom filename (default: extracted_kpis_YYYYMMDD_HHMMSS.json)
        
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"extracted_kpis_{self.timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        return filepath
    
    def save_csv(self, data: List[Dict[str, Any]], filename: str = None) -> Path:
        """
        Save data to CSV file with timestamp.
        
        Args:
            data: List of dictionaries to save
            filename: Optional custom filename
        
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"extracted_kpis_{self.timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Flatten nested dicts for CSV
        df = pd.json_normalize(data, sep="_")
        df.to_csv(filepath, index=False, encoding="utf-8")
        
        return filepath
    
    def load_json(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load data from JSON file.
        
        Args:
            filename: Name of file in output_dir
        
        Returns:
            List of dictionaries
        """
        filepath = self.output_dir / filename
        
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def generate_report(self, stats: Dict[str, Any]) -> Path:
        """
        Generate a summary report of extraction.
        
        Args:
            stats: Dictionary with extraction statistics
        
        Returns:
            Path to report file
        """
        report_file = self.output_dir / f"extraction_report_{self.timestamp}.txt"
        
        with open(report_file, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("CELONIS EXTRACTION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Total KPIs: {stats.get('total', 0)}\n")
            f.write(f"By Type: {stats.get('by_type', {})}\n")
            f.write(f"By Record: {stats.get('by_record', {})}\n")
            f.write("=" * 60 + "\n")
        
        return report_file

    def generate_kpi_breakdown_report(
        self,
        kpi_dicts: List[Dict[str, Any]],
        filename: str = None,
    ) -> Path:
        """
        Generate a readable breakdown report of extracted KPI items.
        Items are separated by `attribute_type` into KPI / Filter / Attribute.
        """
        if filename is None:
            filename = f"kpi_breakdown_report_{self.timestamp}.txt"

        report_file = self.output_dir / filename

        by_type: Dict[str, List[Dict[str, Any]]] = {"KPI": [], "Filter": [], "Attribute": []}
        for item in kpi_dicts:
            t = str(item.get("attribute_type", "") or "")
            # Normalize some common variations
            if t.lower() == "kpi":
                by_type["KPI"].append(item)
            elif t.lower() == "filter":
                by_type["Filter"].append(item)
            elif t.lower() == "attribute":
                by_type["Attribute"].append(item)

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("CELONIS KPI EXTRACTION BREAKDOWN\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write("\n")

            for section in ["KPI", "Filter", "Attribute"]:
                items = by_type[section]
                f.write(f"--- {section} ({len(items)} items) ---\n")
                for it in items:
                    kpi_id = it.get("kpi_id", "") or ""
                    name = it.get("name", "") or ""
                    record_id = it.get("record_id", "") or ""
                    pql = it.get("pql_formula", "") or ""

                    # Avoid extremely long lines, but still keep the core PQL visible.
                    pql_single_line = str(pql).replace("\n", " ").strip()
                    if len(pql_single_line) > 250:
                        pql_single_line = pql_single_line[:247] + "..."

                    f.write(f"- {name} | kpi_id={kpi_id} | record_id={record_id}\n")
                    if pql_single_line:
                        f.write(f"  pql: {pql_single_line}\n")
                f.write("\n")

        return report_file