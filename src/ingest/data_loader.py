"""
Format-agnostic data loader for CSES variable mapping.

Handles: .dta (Stata), .sav (SPSS), .csv, .xlsx, .json, .parquet
Extracts: variable names, descriptions (if available), value labels (if available), sample values
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Optional imports - handle gracefully if not available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    logger.warning("pandas not available - data file reading will be limited")

try:
    import pyreadstat
    PYREADSTAT_AVAILABLE = True
except ImportError:
    PYREADSTAT_AVAILABLE = False
    pyreadstat = None
    logger.warning("pyreadstat not available - cannot read .dta/.sav files directly")


@dataclass
class VariableInfo:
    """Information about a single variable."""
    name: str
    description: Optional[str] = None
    value_labels: Optional[dict] = None
    sample_values: list = field(default_factory=list)
    dtype: Optional[str] = None
    n_unique: Optional[int] = None

    def has_metadata(self) -> bool:
        """Check if variable has any metadata beyond just the name."""
        return bool(self.description or self.value_labels)

    def to_context_string(self) -> str:
        """Convert to string for LLM context."""
        parts = [f"- {self.name}"]

        if self.description:
            parts.append(f': "{self.description}"')

        if self.sample_values:
            samples = self.sample_values[:3]
            parts.append(f" [samples: {samples}]")

        if self.value_labels:
            # Show first few labels
            labels = dict(list(self.value_labels.items())[:3])
            parts.append(f" [labels: {labels}]")

        return "".join(parts)


@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""
    file_path: Path
    file_format: str
    n_rows: int
    n_variables: int
    variables: dict[str, VariableInfo]
    metadata_quality: str  # "rich", "partial", "minimal"
    load_errors: list[str] = field(default_factory=list)

    def get_variables_with_metadata(self) -> list[VariableInfo]:
        """Get variables that have descriptions or labels."""
        return [v for v in self.variables.values() if v.has_metadata()]

    def get_variables_without_metadata(self) -> list[VariableInfo]:
        """Get variables with only names."""
        return [v for v in self.variables.values() if not v.has_metadata()]

    def summary(self) -> str:
        """Get a summary of the dataset."""
        with_meta = len(self.get_variables_with_metadata())
        without_meta = len(self.get_variables_without_metadata())
        return (
            f"File: {self.file_path.name}\n"
            f"Format: {self.file_format}\n"
            f"Rows: {self.n_rows}, Variables: {self.n_variables}\n"
            f"Metadata quality: {self.metadata_quality}\n"
            f"  - With descriptions/labels: {with_meta}\n"
            f"  - Names only: {without_meta}"
        )


class DataLoader:
    """Format-agnostic data loader - supports any tabular format."""

    SUPPORTED_FORMATS = {
        # Statistical software formats
        '.dta': 'stata',
        '.sav': 'spss',
        '.por': 'spss',
        # Spreadsheet formats
        '.csv': 'csv',
        '.tsv': 'csv',
        '.txt': 'csv',  # Try as delimited text
        '.dat': 'csv',  # Try as delimited text
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.ods': 'excel',  # OpenDocument spreadsheet
        # Data interchange formats
        '.json': 'json',
        '.parquet': 'parquet',
        '.pq': 'parquet',
        '.feather': 'feather',
        '.ftr': 'feather',
    }

    def __init__(self):
        self.errors = []

    def load(self, file_path: Path | str) -> Optional[DatasetInfo]:
        """
        Load a data file and extract variable information.
        Attempts to auto-detect format if extension is unknown.

        Args:
            file_path: Path to the data file

        Returns:
            DatasetInfo object or None if loading fails
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        suffix = file_path.suffix.lower()

        # If format unknown, try CSV as fallback (most common)
        if suffix not in self.SUPPORTED_FORMATS:
            logger.warning(f"Unknown format {suffix}, attempting to load as CSV")
            format_type = 'csv'
        else:
            format_type = self.SUPPORTED_FORMATS[suffix]

        try:
            if format_type == 'stata':
                return self._load_stata(file_path)
            elif format_type == 'spss':
                return self._load_spss(file_path)
            elif format_type == 'csv':
                return self._load_csv(file_path)
            elif format_type == 'excel':
                return self._load_excel(file_path)
            elif format_type == 'json':
                return self._load_json(file_path)
            elif format_type == 'parquet':
                return self._load_parquet(file_path)
            elif format_type == 'feather':
                return self._load_feather(file_path)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _load_stata(self, file_path: Path) -> DatasetInfo:
        """Load Stata .dta file with full metadata."""
        if not PYREADSTAT_AVAILABLE:
            raise ImportError(
                "pyreadstat is required to read Stata (.dta) files but is not installed. "
                "This may be due to Python 3.14 compatibility issues. "
                "Try: pip install pyreadstat"
            )

        df, meta = pyreadstat.read_dta(str(file_path))

        variables = {}
        for col in df.columns:
            variables[col] = VariableInfo(
                name=col,
                description=meta.column_names_to_labels.get(col),
                value_labels=meta.variable_value_labels.get(col),
                sample_values=df[col].dropna().head(5).tolist(),
                dtype=str(df[col].dtype),
                n_unique=df[col].nunique()
            )

        # Assess metadata quality
        with_meta = sum(1 for v in variables.values() if v.has_metadata())
        if with_meta > len(variables) * 0.8:
            quality = "rich"
        elif with_meta > len(variables) * 0.3:
            quality = "partial"
        else:
            quality = "minimal"

        return DatasetInfo(
            file_path=file_path,
            file_format="Stata (.dta)",
            n_rows=len(df),
            n_variables=len(df.columns),
            variables=variables,
            metadata_quality=quality
        )

    def _load_spss(self, file_path: Path) -> DatasetInfo:
        """Load SPSS .sav file with full metadata."""
        if not PYREADSTAT_AVAILABLE:
            raise ImportError(
                "pyreadstat is required to read SPSS (.sav) files but is not installed. "
                "This may be due to Python 3.14 compatibility issues. "
                "Try: pip install pyreadstat"
            )

        # Try different encodings - SPSS files can have various encodings
        df = None
        meta = None
        encodings_to_try = [None, 'UTF-8', 'latin-1', 'cp1252', 'cp1250', 'iso-8859-1', 'iso-8859-2']

        for encoding in encodings_to_try:
            try:
                if encoding:
                    df, meta = pyreadstat.read_sav(str(file_path), encoding=encoding)
                else:
                    df, meta = pyreadstat.read_sav(str(file_path))
                break
            except Exception as e:
                if 'encoding' in str(e).lower() or 'codec' in str(e).lower() or 'byte' in str(e).lower():
                    continue
                # For non-encoding errors, try to disable label reading
                try:
                    df, meta = pyreadstat.read_sav(str(file_path), apply_value_formats=False,
                                                   formats_as_category=False)
                    break
                except:
                    continue

        if df is None:
            raise ValueError(f"Could not load SPSS file with any encoding")

        variables = {}
        for col in df.columns:
            # Safely get metadata, handling potential encoding issues in labels
            try:
                description = meta.column_names_to_labels.get(col) if meta else None
            except:
                description = None
            try:
                value_labels = meta.variable_value_labels.get(col) if meta else None
            except:
                value_labels = None

            variables[col] = VariableInfo(
                name=col,
                description=description,
                value_labels=value_labels,
                sample_values=df[col].dropna().head(5).tolist(),
                dtype=str(df[col].dtype),
                n_unique=df[col].nunique()
            )

        with_meta = sum(1 for v in variables.values() if v.has_metadata())
        if with_meta > len(variables) * 0.8:
            quality = "rich"
        elif with_meta > len(variables) * 0.3:
            quality = "partial"
        else:
            quality = "minimal"

        return DatasetInfo(
            file_path=file_path,
            file_format="SPSS (.sav)",
            n_rows=len(df),
            n_variables=len(df.columns),
            variables=variables,
            metadata_quality=quality
        )

    def _load_csv(self, file_path: Path) -> DatasetInfo:
        """Load CSV/TSV/delimited text file (no embedded metadata)."""
        import csv

        # Auto-detect delimiter by sniffing the file
        delimiter = None

        # Try to detect delimiter from file sample
        try:
            with open(file_path, 'rb') as f:
                raw_sample = f.read(8192)
            # Decode sample with latin-1 (never fails)
            sample = raw_sample.decode('latin-1')
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=',;\t|')
                delimiter = dialect.delimiter
            except csv.Error:
                # Fallback: check file extension or use comma
                suffix = file_path.suffix.lower()
                if suffix == '.tsv':
                    delimiter = '\t'
                elif ';' in sample and ',' not in sample:
                    delimiter = ';'
                else:
                    delimiter = ','
        except Exception:
            delimiter = ','

        # Try to load with different encodings - pandas will read the FULL file
        df_full = None
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'cp949', 'euc-kr', 'gbk', 'big5']

        for encoding in encodings_to_try:
            try:
                df_full = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                # If it's not an encoding error, try next encoding anyway
                continue

        # Ultimate fallback: latin-1 with error handling via Python engine
        if df_full is None:
            try:
                df_full = pd.read_csv(file_path, delimiter=delimiter, encoding='latin-1',
                                      on_bad_lines='skip', engine='python')
            except Exception:
                # Last resort: read as bytes and decode
                df_full = pd.read_csv(file_path, delimiter=delimiter, encoding='latin-1')

        if df_full is None or len(df_full.columns) == 0:
            raise ValueError(f"Could not parse CSV file with any encoding")

        # Detect format name
        if delimiter == '\t':
            fmt = "TSV"
        elif delimiter == ';':
            fmt = "CSV (semicolon)"
        elif delimiter == '|':
            fmt = "Pipe-delimited"
        else:
            fmt = "CSV"

        variables = {}
        for col in df_full.columns:
            variables[col] = VariableInfo(
                name=col,
                description=None,  # CSV has no embedded descriptions
                value_labels=None,
                sample_values=df_full[col].dropna().head(5).tolist(),
                dtype=str(df_full[col].dtype),
                n_unique=df_full[col].nunique()
            )

        return DatasetInfo(
            file_path=file_path,
            file_format=fmt,
            n_rows=len(df_full),
            n_variables=len(df_full.columns),
            variables=variables,
            metadata_quality="minimal"  # CSV never has metadata
        )

    def _load_excel(self, file_path: Path) -> DatasetInfo:
        """Load Excel file."""
        df = pd.read_excel(file_path)

        variables = {}
        for col in df.columns:
            variables[col] = VariableInfo(
                name=col,
                description=None,
                value_labels=None,
                sample_values=df[col].dropna().head(5).tolist(),
                dtype=str(df[col].dtype),
                n_unique=df[col].nunique()
            )

        return DatasetInfo(
            file_path=file_path,
            file_format="Excel (.xlsx)",
            n_rows=len(df),
            n_variables=len(df.columns),
            variables=variables,
            metadata_quality="minimal"
        )

    def _load_json(self, file_path: Path) -> DatasetInfo:
        """Load JSON file (assumes array of records)."""
        df = pd.read_json(file_path)

        variables = {}
        for col in df.columns:
            variables[col] = VariableInfo(
                name=col,
                description=None,
                value_labels=None,
                sample_values=df[col].dropna().head(5).tolist(),
                dtype=str(df[col].dtype),
                n_unique=df[col].nunique()
            )

        return DatasetInfo(
            file_path=file_path,
            file_format="JSON",
            n_rows=len(df),
            n_variables=len(df.columns),
            variables=variables,
            metadata_quality="minimal"
        )

    def _load_parquet(self, file_path: Path) -> DatasetInfo:
        """Load Parquet file."""
        df = pd.read_parquet(file_path)

        variables = {}
        for col in df.columns:
            variables[col] = VariableInfo(
                name=col,
                description=None,
                value_labels=None,
                sample_values=df[col].dropna().head(5).tolist(),
                dtype=str(df[col].dtype),
                n_unique=df[col].nunique()
            )

        return DatasetInfo(
            file_path=file_path,
            file_format="Parquet",
            n_rows=len(df),
            n_variables=len(df.columns),
            variables=variables,
            metadata_quality="minimal"
        )

    def _load_feather(self, file_path: Path) -> DatasetInfo:
        """Load Feather/Arrow file."""
        df = pd.read_feather(file_path)

        variables = {}
        for col in df.columns:
            variables[col] = VariableInfo(
                name=col,
                description=None,
                value_labels=None,
                sample_values=df[col].dropna().head(5).tolist(),
                dtype=str(df[col].dtype),
                n_unique=df[col].nunique()
            )

        return DatasetInfo(
            file_path=file_path,
            file_format="Feather",
            n_rows=len(df),
            n_variables=len(df.columns),
            variables=variables,
            metadata_quality="minimal"
        )


def load_data(file_path: Path | str) -> Optional[DatasetInfo]:
    """Convenience function to load a data file."""
    loader = DataLoader()
    return loader.load(file_path)
