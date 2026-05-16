"""
Full pipeline test for SouthKorea_2024 using the new multi-model ensemble architecture.
Compares output to Sweden_2022 reference.
"""

import sys
import json
import logging
import os
from pathlib import Path
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv

# Try multiple locations for .env
env_paths = [
    Path.home() / '.cses-agent' / '.env',
    Path(__file__).parent / '.env',
    Path.cwd() / '.env',
]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment from: {env_path}")
        break

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_full_pipeline():
    """Run the full pipeline on SouthKorea_2024 data."""

    # Define paths
    korea_dir = Path("SouthKorea_2024/emails/20250303")
    sweden_dir = Path("Sweden_2022")
    output_dir = Path("SouthKorea_2024/micro")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Korea input files
    data_file = korea_dir / "5. CSES6 South Korea_Micro Data.csv"
    codebook_file = korea_dir / "7. CSES6 South Korea_Micro Data Codebook.txt"
    questionnaire_file = korea_dir / "4. CSES6 South Korea_English Back Translation.pdf"
    design_report_file = korea_dir / "1. CSES6 South Korea_Design Report.docx"

    # Sweden reference
    sweden_do_file = sweden_dir / "micro" / "cses-m6_micro_SWE_2022_20250116.do"

    print("=" * 70)
    print("CSES PIPELINE TEST - SouthKorea_2024")
    print("=" * 70)
    print()

    # Check files exist
    print("Checking input files...")
    files_to_check = [
        ("Data file", data_file),
        ("Codebook", codebook_file),
        ("Questionnaire", questionnaire_file),
        ("Design report", design_report_file),
        ("Sweden reference", sweden_do_file),
    ]

    all_exist = True
    for name, path in files_to_check:
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"  {name}: {status} - {path}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\nERROR: Some files are missing. Cannot proceed.")
        return False

    print("\nAll input files found.")
    print()

    # =========================================================================
    # STAGE 0: Document Quality Assessment
    # =========================================================================
    print("=" * 70)
    print("STAGE 0: DOCUMENT QUALITY ASSESSMENT")
    print("=" * 70)

    try:
        from src.preprocessing.quality_assessor import DocumentQualityAssessor
        from src.ingest.doc_parser import DocumentParser

        assessor = DocumentQualityAssessor()
        parser = DocumentParser()

        # Load document texts
        print("\nLoading documents...")

        codebook_text = ""
        if codebook_file.exists():
            codebook_text = codebook_file.read_text(encoding='utf-8', errors='replace')
            print(f"  Codebook: {len(codebook_text)} chars")

        questionnaire_text = ""
        if questionnaire_file.exists():
            doc_info = parser.parse(questionnaire_file)
            if doc_info:
                questionnaire_text = doc_info.full_text or ""
                print(f"  Questionnaire: {len(questionnaire_text)} chars")

        design_text = ""
        if design_report_file.exists():
            doc_info = parser.parse(design_report_file)
            if doc_info:
                design_text = doc_info.full_text or ""
                print(f"  Design report: {len(design_text)} chars")

        print("\nAssessing document quality...")
        quality_report = assessor.assess_all_documents(
            codebook_text=codebook_text,
            questionnaire_text=questionnaire_text,
            design_report_text=design_text,
            progress_callback=lambda msg: print(f"  {msg}")
        )

        print("\n" + quality_report.summary())

        # Save quality report
        quality_output = output_dir / "quality_report.json"
        with open(quality_output, 'w') as f:
            json.dump({
                "overall_score": quality_report.overall_score,
                "can_proceed": quality_report.can_proceed,
                "critical_issues": quality_report.critical_issues,
                "documents": {
                    doc_type: {
                        "overall_score": dq.overall_score,
                        "is_usable": dq.is_usable,
                        "issues": dq.issues,
                        "recommendations": dq.recommendations
                    }
                    for doc_type, dq in quality_report.documents.items()
                }
            }, f, indent=2)
        print(f"\nQuality report saved to: {quality_output}")

        if not quality_report.can_proceed:
            print("\nWARNING: Document quality is insufficient, but proceeding anyway for testing...")

    except Exception as e:
        logger.error(f"Quality assessment failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nQuality assessment failed: {e}")
        print("Continuing with pipeline test...")

    print()

    # =========================================================================
    # STAGE 1: Load Data and Extract Context
    # =========================================================================
    print("=" * 70)
    print("STAGE 1: LOAD DATA AND EXTRACT CONTEXT")
    print("=" * 70)

    try:
        from src.ingest.data_loader import DataLoader
        from src.preprocessing.document_aggregator import DocumentAggregator

        print("\nLoading data file...")
        loader = DataLoader()
        dataset_info = loader.load(data_file)

        if dataset_info:
            print(f"  Loaded {len(dataset_info.variables)} variables, {dataset_info.n_rows} rows")

            # Get source contexts - variables is a dict of name -> VariableInfo
            source_contexts = []
            for var_name, var_info in dataset_info.variables.items():
                ctx = {
                    'name': var_info.name,
                    'description': var_info.description or '',
                    'dtype': str(var_info.dtype) if var_info.dtype else '',
                    'value_labels': var_info.value_labels or {},
                    'sample_values': var_info.sample_values or [],
                    'n_unique': var_info.n_unique,
                }
                source_contexts.append(ctx)

            print(f"  Extracted context for {len(source_contexts)} variables")

            # Save source contexts
            contexts_output = output_dir / "source_contexts.json"
            with open(contexts_output, 'w') as f:
                json.dump(source_contexts, f, indent=2, default=str)
            print(f"  Source contexts saved to: {contexts_output}")
        else:
            print("  ERROR: Failed to load data file")
            return False

    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # =========================================================================
    # STAGE 1.5: Document Aggregation (TOON Summary)
    # =========================================================================
    print("=" * 70)
    print("STAGE 1.5: DOCUMENT AGGREGATION (TOON SUMMARY)")
    print("=" * 70)

    pre_aggregated_summary = ""
    try:
        aggregator = DocumentAggregator()

        # Collect all document texts
        doc_texts = []
        if codebook_text:
            doc_texts.append(("codebook", codebook_text))
        if questionnaire_text:
            doc_texts.append(("questionnaire", questionnaire_text))

        print("\nAggregating documents into TOON format...")
        pre_aggregated_summary = aggregator.aggregate_variable_info(
            source_variables=source_contexts,
            codebook_text=codebook_text,
            questionnaire_text=questionnaire_text,
            design_report_text=design_text,
            progress_callback=lambda msg: print(f"  {msg}")
        )

        print(f"\nTOON summary generated: {len(pre_aggregated_summary)} chars")

        # Save TOON summary
        toon_output = output_dir / "toon_summary.txt"
        with open(toon_output, 'w', encoding='utf-8') as f:
            f.write(pre_aggregated_summary)
        print(f"TOON summary saved to: {toon_output}")

    except Exception as e:
        logger.error(f"Document aggregation failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"Document aggregation failed: {e}")
        print("Will use basic source contexts for matching...")

        # Fallback: create basic TOON from source contexts
        from src.utils.toon_encoder import encode_table
        rows = [{'name': c['name'], 'desc': c['description'][:100], 'labels': str(c.get('value_labels', {}))[:100]}
                for c in source_contexts[:50]]
        pre_aggregated_summary = 'source_vars' + encode_table(rows, ['name', 'desc', 'labels'], '|')

    print()

    # =========================================================================
    # STAGE 2: ENSEMBLE MATCHING (Multi-Model Quorum)
    # =========================================================================
    print("=" * 70)
    print("STAGE 2: ENSEMBLE MATCHING (Multi-Model Quorum)")
    print("=" * 70)

    mappings = []
    try:
        from src.matching.ensemble_matcher import EnsembleMatcher

        print("\nInitializing ensemble matcher...")
        matcher = EnsembleMatcher()
        print(f"  Models: {matcher.models}")
        print(f"  Quorum threshold: {matcher.quorum_threshold}")

        print("\nRunning ensemble matching (this may take a few minutes)...")
        result = matcher.match_with_quorum(
            source_contexts=source_contexts,
            pre_aggregated_summary=pre_aggregated_summary,
            progress_callback=lambda msg: print(f"  {msg}")
        )

        # Convert to list of dicts
        mappings = []
        for proposal in result.proposals:
            mappings.append({
                "target": proposal.target_variable,
                "source": proposal.source_variable,
                "confidence": proposal.confidence,
                "confidence_level": proposal.confidence_level,
                "reasoning": proposal.reasoning,
                "needs_review": proposal.needs_review
            })

        # Summary
        matched = len([m for m in mappings if m["source"] not in ["NOT_FOUND", "ERROR", "NO_CONSENSUS"]])
        high_conf = len([m for m in mappings if m["confidence_level"] == "high"])
        needs_review = len([m for m in mappings if m["needs_review"]])

        print(f"\nMatching Results:")
        print(f"  Total targets: {len(mappings)}")
        print(f"  Matched: {matched}")
        print(f"  High confidence: {high_conf}")
        print(f"  Needs review: {needs_review}")
        print(f"  Not found: {len(mappings) - matched}")

        # Save mappings
        mappings_output = output_dir / "ensemble_mappings.json"
        with open(mappings_output, 'w') as f:
            json.dump(mappings, f, indent=2)
        print(f"\nMappings saved to: {mappings_output}")

    except Exception as e:
        logger.error(f"Ensemble matching failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nEnsemble matching failed: {e}")
        print("Falling back to single-model matching...")

        try:
            from src.matching.llm_matcher import LLMMatcher
            matcher = LLMMatcher()
            result = matcher.match_variables(
                source_contexts=source_contexts,
                pre_aggregated_summary=pre_aggregated_summary,
                progress_callback=lambda msg: print(f"  {msg}")
            )

            mappings = []
            for proposal in result.proposals:
                mappings.append({
                    "target": proposal.target_variable,
                    "source": proposal.source_variable,
                    "confidence": proposal.confidence,
                    "confidence_level": proposal.confidence_level,
                    "reasoning": proposal.reasoning,
                    "needs_review": getattr(proposal, 'needs_review', False)
                })

            print(f"\nSingle-model matching complete: {len([m for m in mappings if m['source'] not in ['NOT_FOUND', 'ERROR']])} matched")

        except Exception as e2:
            logger.error(f"Single-model matching also failed: {e2}")
            import traceback
            traceback.print_exc()
            return False

    print()

    # =========================================================================
    # STAGE 3: RECODING STRATEGY GENERATION
    # =========================================================================
    print("=" * 70)
    print("STAGE 3: RECODING STRATEGY GENERATION")
    print("=" * 70)

    recoding_strategies = {}
    try:
        from src.matching.recoding_strategist import RecodingStrategist

        print("\nGenerating recoding strategies...")
        strategist = RecodingStrategist()

        recoding_strategies = strategist.generate_strategies(
            mappings=mappings,
            source_contexts=source_contexts,
            progress_callback=lambda msg: print(f"  {msg}")
        )

        print(f"\nGenerated {len(recoding_strategies)} recoding strategies")

        # Summary by type
        by_type = {}
        for strategy in recoding_strategies.values():
            t = strategy.transformation_type
            by_type[t] = by_type.get(t, 0) + 1

        print("  By transformation type:")
        for t, count in sorted(by_type.items()):
            print(f"    {t}: {count}")

        # Save strategies
        strategies_output = output_dir / "recoding_strategies.json"
        with open(strategies_output, 'w') as f:
            json.dump({k: v.to_dict() for k, v in recoding_strategies.items()}, f, indent=2)
        print(f"\nRecoding strategies saved to: {strategies_output}")

    except Exception as e:
        logger.error(f"Recoding strategy generation failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nRecoding strategy generation failed: {e}")
        print("Continuing without recoding strategies...")

    print()

    # =========================================================================
    # STAGE 4: STATA CODE GENERATION (with Sweden patterns)
    # =========================================================================
    print("=" * 70)
    print("STAGE 4: STATA CODE GENERATION (with Sweden patterns)")
    print("=" * 70)

    try:
        from src.agent.tool_wrappers import generate_do_file_with_llm
        from src.reference.sweden_patterns import load_sweden_patterns

        # Load Sweden patterns
        print("\nLoading Sweden reference patterns...")
        sweden_patterns = load_sweden_patterns()
        print(f"  Loaded {len(sweden_patterns.variables)} variable patterns from Sweden")

        print("\nGenerating .do file...")
        do_content = generate_do_file_with_llm(
            mappings=mappings,
            source_contexts=source_contexts,
            country_code="KOR",
            country_name="South Korea",
            year="2024",
            author="CSES Pipeline Test",
            data_file_path=str(data_file.absolute()),
            party_result=None,
            study_number=1,
            progress_callback=lambda msg: print(f"  {msg}"),
            recoding_strategies=recoding_strategies
        )

        # Save .do file
        do_output = output_dir / f"cses-m6_micro_KOR_2024_{datetime.now().strftime('%Y%m%d')}.do"
        with open(do_output, 'w', encoding='utf-8') as f:
            f.write(do_content)
        print(f"\n.do file saved to: {do_output}")
        print(f"  Size: {len(do_content)} chars, {do_content.count(chr(10))} lines")

    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nCode generation failed: {e}")
        return False

    print()

    # =========================================================================
    # STAGE 5: COMPARE TO SWEDEN REFERENCE
    # =========================================================================
    print("=" * 70)
    print("STAGE 5: COMPARE TO SWEDEN REFERENCE")
    print("=" * 70)

    try:
        sweden_content = sweden_do_file.read_text(encoding='utf-8', errors='replace')

        print("\nComparison:")
        print(f"  Sweden .do file: {len(sweden_content)} chars, {sweden_content.count(chr(10))} lines")
        print(f"  Korea .do file:  {len(do_content)} chars, {do_content.count(chr(10))} lines")

        # Check for key patterns
        patterns_to_check = [
            ("Section headers (\\\\\\)", r"\*\*\\\\\\"),
            ("Variable headers (>>>)", ">>>"),
            ("Tab verification", "tab .*, mis"),
            ("Gen commands", "gen "),
            ("Recode commands", "recode "),
            ("Replace commands", "replace "),
        ]

        import re
        print("\nPattern comparison:")
        print(f"  {'Pattern':<30} {'Sweden':<10} {'Korea':<10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10}")

        for name, pattern in patterns_to_check:
            sweden_count = len(re.findall(pattern, sweden_content))
            korea_count = len(re.findall(pattern, do_content))
            print(f"  {name:<30} {sweden_count:<10} {korea_count:<10}")

    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        print(f"\nComparison failed: {e}")

    print()
    print("=" * 70)
    print("PIPELINE TEST COMPLETE")
    print("=" * 70)
    print()
    print("Output files:")
    for f in output_dir.glob("*"):
        if f.is_file():
            print(f"  {f.name}")

    return True


if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
