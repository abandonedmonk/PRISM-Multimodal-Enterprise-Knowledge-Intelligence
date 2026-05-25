""" ╔════════════════════════════════════════════════════════════════════════════╗
    ║  Table Image Renderer                                                       ║
    ║  Renders HTML table elements to PNG images using Playwright.              ║
    ╚════════════════════════════════════════════════════════════════════════════╝

Purpose:
    Convert HTML table elements from SEC filings into clean PNG images for
    vision-based retrieval and VLM description generation.

Usage:
    from ingestion.core.table_renderer import render_table_to_image

    image_path = render_table_to_image(
        raw_html="<table><tr><td>...</td></tr></table>",
        output_dir="data/processed/tables",
        filename="NVDA_2024_10K_item7_table3.png",
        width=1200
    )
"""

import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CSS_WRAPPER = """
<!DOCTYPE html>
<html>
<head>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    background: #ffffff;
    padding: 20px;
    font-size: 13px;
    color: #1a1a1a;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    max-width: {width}px;
    table-layout: fixed;
  }
  th {
    background: #f5f5f5;
    color: #1a1a1a;
    font-weight: 600;
    text-align: left;
    padding: 10px 12px;
    border: 1px solid #d0d0d0;
    font-size: 12px;
    letter-spacing: 0.3px;
  }
  td {
    padding: 8px 12px;
    border: 1px solid #e0e0e0;
    vertical-align: top;
    word-wrap: break-word;
    overflow-wrap: break-word;
    font-size: 12px;
    line-height: 1.5;
  }
  tr:nth-child(even) td {
    background: #fafafa;
  }
  tr:hover td {
    background: #f0f7ff;
  }
  .numeric {
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
</style>
</head>
<body>{table_html}</body>
</html>
"""


def _get_playwright_browser():
    try:
        from playwright.sync_api import sync_playwright

        return sync_playwright().start().chromium
    except ImportError:
        raise ImportError(
            "playwright not installed. Run: playwright install chromium"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to start Playwright: {e}")


def render_table_to_image(
    raw_html: str,
    output_dir: str | Path,
    filename: str,
    width: int = 1200,
) -> str:
    """Render an HTML table to a PNG image using Playwright headless Chromium.

    Args:
        raw_html: Raw HTML string of the table element
        output_dir: Directory to save the PNG image
        filename: Name of the output PNG file
        width: Render width in pixels (default 1200)

    Returns:
        Absolute path to the saved PNG image, or empty string on failure
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    if not raw_html or not raw_html.strip():
        return ""

    wrapped_html = CSS_WRAPPER.format(
        table_html=raw_html, width=width
    )

    browser = None
    page = None
    try:
        browser = _get_playwright_browser()
        page = browser.new_page()

        page.set_content(wrapped_html, wait_until="networkidle")
        page.wait_for_selector("table", timeout=5000)

        viewport = {"width": width + 40, "height": 800}
        page.set_viewport_size(viewport)

        page.screenshot(
            path=str(output_path),
            type="png",
            full_page=True,
        )
        return str(output_path.resolve())

    except Exception as e:
        print(f"  WARNING: Failed to render table to image: {e}")
        return ""
    finally:
        if page:
            page.close()
        if browser:
            browser.contexts[0].browser.close()


def install_playwright():
    import subprocess

    subprocess.run(
        ["playwright", "install", "chromium"],
        check=True,
    )
    print("Playwright Chromium installed successfully.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Install Playwright Chromium browser")
    parser.add_argument(
        "--yes", action="store_true", help="Skip confirmation prompt"
    )
    args = parser.parse_args()

    install_playwright()