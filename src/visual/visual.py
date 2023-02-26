from pathlib import Path

import imageio
from PIL import Image
from playwright.sync_api import sync_playwright

from src.logger import get_logger

logger = get_logger()


def html2ani(source_path: Path, save_file: Path) -> None:
    temp = Path().cwd() / "temp"
    logger.info("Converting HTML to PNG")
    html_files = source_path.glob("*.html")
    for html in html_files:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False, args=["--start-maximized"])
            page = browser.new_page()
            page.goto(
                f"file://{html}",
                timeout=60000,
                wait_until="networkidle",
            )
            temp_file = temp / f"{html.stem}.png"
            page.screenshot(path=temp_file, full_page=True)
            browser.close()
    logger.info("Converting PNG to animated GIF")
    png_files = [str(path) for path in temp.glob("*.png")]
    pngs = [Image.open(png) for png in sorted(png_files)]
    imageio.mimsave(save_file, pngs, "GIF", fps=2)
    logger.info("finished.")


if __name__ == "__main__":
    html_path = Path.cwd() / "data" / "visual" / "html"
    save_file = Path.cwd() / "data" / "visual" / "ani" / "ani.gif"
    html2ani(html_path, save_file)
