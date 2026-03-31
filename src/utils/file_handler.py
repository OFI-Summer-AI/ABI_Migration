"""
File handling utilities for JSON and CSV export.
"""
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd

# FileHandler class to handle saving and loading of extraction data.
class FileHandler:
    """
    Handles saving and loading of extraction data.
    """
    # Initialize with an output directory and timestamp.
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save data to JSON file with timestamp.
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
    
    # Save data to CSV file with timestamp.
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

    # Save a pandas DataFrame to CSV.
    def save_dataframe(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Save a pandas DataFrame to CSV.
        
        Args:
            df: DataFrame to save
            filename: Target filename
            
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False, encoding="utf-8")
        return filepath
    
    # Load data from JSON file.
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
    
    # Generate a summary report of extraction.
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