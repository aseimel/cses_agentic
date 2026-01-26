"""
File Organizer for CSES Processing.

Detects collaborator files and organizes them into a single clean folder
with all files well-named using the country_year prefix.
"""

import logging
import shutil
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# File type detection patterns
DATA_EXTENSIONS = {".dta", ".sav", ".csv", ".xlsx", ".xls", ".json", ".parquet"}
DOC_EXTENSIONS = {".docx", ".doc", ".pdf", ".txt", ".rtf"}

# Keywords for file type identification
QUESTIONNAIRE_KEYWORDS = [
    "questionnaire", "fragebogen", "cuestionario", "questionario",
    "survey", "translation", "pregunta", "enquete", "encuesta"
]
CODEBOOK_KEYWORDS = [
    "codebook", "code book", "variable", "dictionary", "codebuch",
    "diccionario", "dicionario"
]
DESIGN_KEYWORDS = [
    "design", "methodology", "technical", "rapport", "bericht", "informe"
]
MACRO_REPORT_KEYWORDS = [
    "macroreport", "macro report", "macro_report", "macro-report",
    "macro data", "macro_data", "macrodata"
]

# Keywords for district data (election results by constituency)
DISTRICT_DATA_KEYWORDS = [
    "district", "constituency", "riding", "wahlkreis", "circonscription"
]

# Keywords to identify English translation questionnaires
ENGLISH_KEYWORDS = [
    "english", "eng", "_en_", "_en.", "(en)", "[en]",
    "translation", "translated", "engl"
]

# Preferred format order for documents (first = most preferred)
DOC_FORMAT_PREFERENCE = [".pdf", ".docx", ".doc", ".rtf", ".txt"]

# Preferred format order for data files (first = most preferred)
DATA_FORMAT_PREFERENCE = [".dta", ".sav", ".parquet", ".csv", ".xlsx", ".xls", ".json"]


def filename_similarity(name1: str, name2: str) -> float:
    """
    Calculate similarity between two filenames (without extensions).
    Uses Jaccard similarity on character trigrams.

    Returns:
        Float between 0 and 1, where 1 = identical
    """
    def get_trigrams(s: str) -> set:
        s = s.lower()
        if len(s) < 3:
            return {s}
        return {s[i:i+3] for i in range(len(s) - 2)}

    trigrams1 = get_trigrams(name1)
    trigrams2 = get_trigrams(name2)

    if not trigrams1 or not trigrams2:
        return 0.0

    intersection = len(trigrams1 & trigrams2)
    union = len(trigrams1 | trigrams2)

    return intersection / union if union > 0 else 0.0


def deduplicate_by_format(
    files: list[Path],
    format_preference: list[str] = None,
    similarity_threshold: float = 0.7
) -> list[Path]:
    """
    Remove duplicate files that exist in multiple formats.
    Keeps the preferred format based on the preference list.

    Only deduplicates if filenames are similar enough (above threshold).
    All originals are still preserved in original_deposit/.

    Args:
        files: List of file paths
        format_preference: Ordered list of preferred extensions (first = most preferred)
                          Defaults to DOC_FORMAT_PREFERENCE
        similarity_threshold: Minimum similarity to consider files as duplicates

    Returns:
        Deduplicated list with preferred formats
    """
    if len(files) <= 1:
        return files

    if format_preference is None:
        format_preference = DOC_FORMAT_PREFERENCE

    # Group files by their base name similarity
    groups = []  # List of lists of similar files
    used = set()

    for i, file1 in enumerate(files):
        if i in used:
            continue

        group = [file1]
        used.add(i)
        stem1 = file1.stem

        for j, file2 in enumerate(files):
            if j in used:
                continue
            stem2 = file2.stem

            if filename_similarity(stem1, stem2) >= similarity_threshold:
                group.append(file2)
                used.add(j)

        groups.append(group)

    # For each group, pick the preferred format
    result = []
    for group in groups:
        if len(group) == 1:
            result.append(group[0])
        else:
            # Sort by format preference
            def format_rank(f: Path) -> int:
                ext = f.suffix.lower()
                try:
                    return format_preference.index(ext)
                except ValueError:
                    return len(format_preference)

            group.sort(key=format_rank)
            preferred = group[0]

            # Log what we're deduplicating
            others = [f.name for f in group[1:]]
            logger.info(f"Deduplicating: keeping {preferred.name}, skipping {others}")

            result.append(preferred)

    return result


@dataclass
class DetectedFiles:
    """Results of file detection in a directory."""
    data_files: list[Path] = field(default_factory=list)  # Survey data (main)
    district_data_files: list[Path] = field(default_factory=list)  # District/constituency election results
    questionnaire_files: list[Path] = field(default_factory=list)
    codebook_files: list[Path] = field(default_factory=list)
    design_report_files: list[Path] = field(default_factory=list)
    macro_report_files: list[Path] = field(default_factory=list)
    other_files: list[Path] = field(default_factory=list)

    # Detected country/year (if identifiable)
    country: Optional[str] = None
    country_code: Optional[str] = None
    year: Optional[str] = None

    def has_minimum_requirements(self) -> bool:
        """Check if minimum required files are present."""
        return len(self.data_files) > 0 and len(self.questionnaire_files) > 0

    def summary(self) -> str:
        """Generate summary of detected files."""
        lines = ["## Detected Files"]

        if self.country and self.year:
            lines.append(f"\nIdentified study: **{self.country} {self.year}**")

        lines.extend([
            "",
            f"### Survey Data ({len(self.data_files)})"
        ])
        for f in self.data_files:
            lines.append(f"  - {f.name}")

        if self.district_data_files:
            lines.extend([
                "",
                f"### District Data ({len(self.district_data_files)})"
            ])
            for f in self.district_data_files:
                lines.append(f"  - {f.name}")

        lines.extend([
            "",
            f"### Questionnaires ({len(self.questionnaire_files)})"
        ])
        for f in self.questionnaire_files:
            lines.append(f"  - {f.name}")

        if self.codebook_files:
            lines.extend([
                "",
                f"### Codebooks ({len(self.codebook_files)})"
            ])
            for f in self.codebook_files:
                lines.append(f"  - {f.name}")

        if self.design_report_files:
            lines.extend([
                "",
                f"### Design Reports ({len(self.design_report_files)})"
            ])
            for f in self.design_report_files:
                lines.append(f"  - {f.name}")

        if self.macro_report_files:
            lines.extend([
                "",
                f"### Macro Reports ({len(self.macro_report_files)})"
            ])
            for f in self.macro_report_files:
                lines.append(f"  - {f.name}")

        if self.other_files:
            lines.extend([
                "",
                f"### Other files ({len(self.other_files)})"
            ])
            for f in self.other_files[:10]:
                lines.append(f"  - {f.name}")
            if len(self.other_files) > 10:
                lines.append(f"  ... and {len(self.other_files) - 10} more")

        # Status
        lines.extend(["", "### Status"])
        if self.has_minimum_requirements():
            lines.append("Ready to process (data + questionnaire found)")
        else:
            lines.append("Missing required files:")
            if not self.data_files:
                lines.append("  - No data file detected")
            if not self.questionnaire_files:
                lines.append("  - No questionnaire detected")

        return "\n".join(lines)


def detect_questionnaire_language(filename: str) -> str:
    """
    Detect if a questionnaire is the English translation or native version.

    Returns:
        'english' or 'native'
    """
    name_lower = filename.lower()

    if any(kw in name_lower for kw in ENGLISH_KEYWORDS):
        return "english"

    return "native"


class FileOrganizer:
    """
    Detects and organizes collaborator files into a clean folder structure.

    Creates a study folder with original deposit preserved:

    KOR_2024/
    ├── original_deposit/                 # Untouched original files
    │   ├── Korea_Survey_Data.dta
    │   ├── CSES_M6_Korea_English.pdf
    │   └── Korea_Questionnaire.pdf
    │
    ├── KOR_2024_data.dta                 # Working copy of data
    ├── KOR_2024_questionnaire_english.pdf
    ├── KOR_2024_questionnaire_native.pdf
    ├── KOR_2024_codebook.docx
    ├── KOR_2024_design_report.pdf
    ├── KOR_2024_macro_report.docx
    │
    │   --- Generated by processing ---
    ├── KOR_2024_variable_mappings.xlsx
    ├── KOR_2024_tracking_sheet.xlsx
    ├── KOR_2024_processing.do
    ├── KOR_2024_M6.dta                   # Final harmonized data
    │
    └── .cses/                            # Agent state (hidden)
        └── state.json
    """

    # Country code mapping
    COUNTRY_CODES = {
        "australia": "AUS", "austria": "AUT", "belgium": "BEL",
        "brazil": "BRA", "bulgaria": "BGR", "canada": "CAN",
        "chile": "CHL", "costa rica": "CRI", "croatia": "HRV",
        "czech": "CZE", "czechia": "CZE", "denmark": "DNK",
        "estonia": "EST", "finland": "FIN", "france": "FRA",
        "germany": "DEU", "great britain": "GBR", "uk": "GBR",
        "united kingdom": "GBR", "greece": "GRC", "hungary": "HUN",
        "iceland": "ISL", "ireland": "IRL", "israel": "ISR",
        "italy": "ITA", "japan": "JPN", "korea": "KOR",
        "south korea": "KOR", "latvia": "LVA", "lithuania": "LTU",
        "mexico": "MEX", "netherlands": "NLD", "new zealand": "NZL",
        "norway": "NOR", "peru": "PER", "philippines": "PHL",
        "poland": "POL", "portugal": "PRT", "romania": "ROU",
        "serbia": "SRB", "slovakia": "SVK", "slovenia": "SVN",
        "south africa": "ZAF", "spain": "ESP", "sweden": "SWE",
        "switzerland": "CHE", "taiwan": "TWN", "thailand": "THA",
        "turkey": "TUR", "united states": "USA", "usa": "USA",
        "uruguay": "URY"
    }

    def __init__(self, working_dir: Path = None):
        """Initialize organizer for a directory."""
        self.working_dir = working_dir or Path.cwd()

    def detect_files(self) -> DetectedFiles:
        """Detect and classify files in the working directory."""
        result = DetectedFiles()

        for item in self.working_dir.iterdir():
            if item.name.startswith("."):
                continue
            if item.is_dir():
                continue

            self._classify_file(item, result)

        self._detect_study_info(result)
        return result

    def _classify_file(self, file_path: Path, result: DetectedFiles):
        """Classify a single file by type."""
        ext = file_path.suffix.lower()
        name_lower = file_path.name.lower()

        if ext in DATA_EXTENSIONS:
            # Check if it's district data (election results by constituency)
            if any(kw in name_lower for kw in DISTRICT_DATA_KEYWORDS):
                result.district_data_files.append(file_path)
            else:
                result.data_files.append(file_path)
            return

        if ext in DOC_EXTENSIONS:
            if any(kw in name_lower for kw in QUESTIONNAIRE_KEYWORDS):
                result.questionnaire_files.append(file_path)
            elif any(kw in name_lower for kw in CODEBOOK_KEYWORDS):
                result.codebook_files.append(file_path)
            elif any(kw in name_lower for kw in MACRO_REPORT_KEYWORDS):
                result.macro_report_files.append(file_path)
            elif any(kw in name_lower for kw in DESIGN_KEYWORDS):
                result.design_report_files.append(file_path)
            else:
                result.other_files.append(file_path)
            return

        result.other_files.append(file_path)

    def _detect_study_info(self, result: DetectedFiles):
        """Try to detect country and year from filenames."""
        all_files = (
            result.data_files +
            result.district_data_files +
            result.questionnaire_files +
            result.codebook_files +
            result.design_report_files +
            result.macro_report_files
        )

        for file_path in all_files:
            name = file_path.stem.lower()

            year_match = re.search(r'(202[0-9])', name)
            if year_match and not result.year:
                result.year = year_match.group(1)

            code_match = re.search(r'\b([A-Z]{3})\b', file_path.stem)
            if code_match:
                code = code_match.group(1)
                for country, c_code in self.COUNTRY_CODES.items():
                    if c_code == code:
                        result.country_code = code
                        result.country = country.title()
                        break

            for country, code in self.COUNTRY_CODES.items():
                if country in name:
                    result.country = country.title()
                    result.country_code = code
                    break

    def get_study_prefix(self, country: str, year: str) -> str:
        """Get the standard prefix for file naming (e.g., KOR_2024)."""
        country_code = self.COUNTRY_CODES.get(country.lower(), country[:3].upper())
        return f"{country_code}_{year}"

    def create_study_folder(
        self,
        country: str,
        year: str,
        base_dir: Path = None
    ) -> Path:
        """
        Create the study folder.

        Args:
            country: Country name
            year: Election year
            base_dir: Base directory (default: working_dir)

        Returns:
            Path to created study folder
        """
        if base_dir is None:
            base_dir = self.working_dir

        prefix = self.get_study_prefix(country, year)
        study_dir = base_dir / prefix

        study_dir.mkdir(parents=True, exist_ok=True)
        (study_dir / ".cses").mkdir(exist_ok=True)

        logger.info(f"Created study folder: {study_dir}")
        return study_dir

    def organize_files(
        self,
        detected: DetectedFiles,
        study_dir: Path,
        prefix: str
    ) -> dict:
        """
        Organize and rename files with proper naming convention.

        Creates:
        - original_deposit/ subfolder with untouched copies of all originals
        - Working copies with standardized names in study_dir

        Args:
            detected: DetectedFiles result
            study_dir: Target study directory
            prefix: File prefix (e.g., KOR_2024)

        Returns:
            Dict mapping original paths to new paths
        """
        mapping = {}

        # Create original_deposit subfolder
        original_deposit = study_dir / "original_deposit"
        original_deposit.mkdir(exist_ok=True)

        # Collect all source files for original deposit
        all_source_files = (
            detected.data_files +
            detected.district_data_files +
            detected.questionnaire_files +
            detected.codebook_files +
            detected.design_report_files +
            detected.macro_report_files
        )

        # Copy all originals to original_deposit (untouched)
        for src in all_source_files:
            dst = original_deposit / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                logger.info(f"Preserved original: {src.name}")

        # Pick the single best data file (prefer .dta > .sav > etc.)
        # There should only be one data submission per study
        if detected.data_files:
            def data_format_rank(f: Path) -> int:
                ext = f.suffix.lower()
                try:
                    return DATA_FORMAT_PREFERENCE.index(ext)
                except ValueError:
                    return len(DATA_FORMAT_PREFERENCE)

            # Sort by format preference and pick the best one
            sorted_data = sorted(detected.data_files, key=data_format_rank)
            best_data = sorted_data[0]

            if len(detected.data_files) > 1:
                skipped = [f.name for f in sorted_data[1:]]
                logger.info(f"Data files: using {best_data.name}, skipping {skipped}")

            new_name = f"{prefix}_data{best_data.suffix}"
            dst = study_dir / new_name
            if not dst.exists():
                shutil.copy2(best_data, dst)
                mapping[str(best_data)] = str(dst)
                logger.info(f"Copied: {best_data.name} -> {new_name}")

        # Copy and rename district data (election results by constituency)
        # Used in Step 9 for merging
        if detected.district_data_files:
            district_files = deduplicate_by_format(
                detected.district_data_files,
                format_preference=DATA_FORMAT_PREFERENCE
            )
            for i, src in enumerate(district_files):
                if len(district_files) == 1:
                    new_name = f"{prefix}_district_data{src.suffix}"
                else:
                    new_name = f"{prefix}_district_data_{i+1}{src.suffix}"
                dst = study_dir / new_name
                if not dst.exists():
                    shutil.copy2(src, dst)
                    mapping[str(src)] = str(dst)
                    logger.info(f"Copied: {src.name} -> {new_name}")

        # Copy and rename questionnaires (english vs native)
        # First separate by language, then deduplicate each group
        english_files = []
        native_files = []
        for src in detected.questionnaire_files:
            if detect_questionnaire_language(src.name) == "english":
                english_files.append(src)
            else:
                native_files.append(src)

        # Deduplicate each language group (prefer PDF)
        english_files = deduplicate_by_format(english_files)
        native_files = deduplicate_by_format(native_files)

        all_questionnaires = english_files + native_files

        if len(all_questionnaires) == 1:
            src = all_questionnaires[0]
            new_name = f"{prefix}_questionnaire{src.suffix}"
            dst = study_dir / new_name
            if not dst.exists():
                shutil.copy2(src, dst)
                mapping[str(src)] = str(dst)
                logger.info(f"Copied: {src.name} -> {new_name}")
        else:
            # Copy english questionnaires
            for i, src in enumerate(english_files):
                if len(english_files) == 1:
                    new_name = f"{prefix}_questionnaire_english{src.suffix}"
                else:
                    new_name = f"{prefix}_questionnaire_english_{i+1}{src.suffix}"
                dst = study_dir / new_name
                if not dst.exists():
                    shutil.copy2(src, dst)
                    mapping[str(src)] = str(dst)
                    logger.info(f"Copied: {src.name} -> {new_name}")

            # Copy native questionnaires
            for i, src in enumerate(native_files):
                if len(native_files) == 1:
                    new_name = f"{prefix}_questionnaire_native{src.suffix}"
                else:
                    new_name = f"{prefix}_questionnaire_native_{i+1}{src.suffix}"
                dst = study_dir / new_name
                if not dst.exists():
                    shutil.copy2(src, dst)
                    mapping[str(src)] = str(dst)
                    logger.info(f"Copied: {src.name} -> {new_name}")

        # Copy and rename codebooks (deduplicate first)
        codebooks = deduplicate_by_format(detected.codebook_files)
        for i, src in enumerate(codebooks):
            if len(codebooks) == 1:
                new_name = f"{prefix}_codebook{src.suffix}"
            else:
                new_name = f"{prefix}_codebook_{i+1}{src.suffix}"
            dst = study_dir / new_name
            if not dst.exists():
                shutil.copy2(src, dst)
                mapping[str(src)] = str(dst)
                logger.info(f"Copied: {src.name} -> {new_name}")

        # Copy and rename design reports (deduplicate first)
        design_reports = deduplicate_by_format(detected.design_report_files)
        for i, src in enumerate(design_reports):
            if len(design_reports) == 1:
                new_name = f"{prefix}_design_report{src.suffix}"
            else:
                new_name = f"{prefix}_design_report_{i+1}{src.suffix}"
            dst = study_dir / new_name
            if not dst.exists():
                shutil.copy2(src, dst)
                mapping[str(src)] = str(dst)
                logger.info(f"Copied: {src.name} -> {new_name}")

        # Copy and rename macro reports (deduplicate first)
        macro_reports = deduplicate_by_format(detected.macro_report_files)
        for i, src in enumerate(macro_reports):
            if len(macro_reports) == 1:
                new_name = f"{prefix}_macro_report{src.suffix}"
            else:
                new_name = f"{prefix}_macro_report_{i+1}{src.suffix}"
            dst = study_dir / new_name
            if not dst.exists():
                shutil.copy2(src, dst)
                mapping[str(src)] = str(dst)
                logger.info(f"Copied: {src.name} -> {new_name}")

        return mapping

    def initialize_study(
        self,
        country: str = None,
        year: str = None,
        copy_files: bool = True
    ) -> Tuple[Path, DetectedFiles]:
        """
        Full initialization: detect files, create folder, organize with proper names.

        Returns:
            Tuple of (study_dir, detected_files)
        """
        detected = self.detect_files()

        country = country or detected.country or "Unknown"
        year = year or detected.year or "0000"

        logger.info(f"Initializing study: {country} {year}")

        study_dir = self.create_study_folder(country, year)
        prefix = self.get_study_prefix(country, year)

        if detected.data_files or detected.questionnaire_files:
            self.organize_files(detected, study_dir, prefix)

        return study_dir, detected


def detect_and_summarize(directory: Path = None) -> str:
    """Quick detection and summary for a directory."""
    organizer = FileOrganizer(directory)
    detected = organizer.detect_files()
    return detected.summary()
