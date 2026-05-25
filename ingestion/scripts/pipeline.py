""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Pipeline Entrypoint                                                       ║
    ║  Main CLI entry for process_all_filings orchestrator.                      ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    Convenience entrypoint for the main ingestion pipeline orchestrator.
    Delegates to ingestion.scripts.process_all_filings.

Usage:
    # Process all filings in data/raw/ -> data/processed/
    python -m ingestion.scripts.pipeline
    
    # Or run directly
    python ingestion/scripts/pipeline.py
    
    # With options (passed through to process_all_filings)
    python -m ingestion.scripts.pipeline --force
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import and delegate to main orchestrator
from ingestion.scripts.process_all_filings import main


if __name__ == "__main__":
    main()
