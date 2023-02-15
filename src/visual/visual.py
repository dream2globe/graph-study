from pathlib import Path

import imageio
from PIL import Image
from playwright.sync_api import sync_playwright


def html2ani(source, target, temp=None):
    if temp is None:
        temp = Path().cwd() / "data" / "visual" / "temp"
    html_files = source.glob("*.html")
    for html in html_files:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False, args=["--start-maximized"])
            page = browser.new_page()
            page.goto(
                f"file://{html}",
                timeout=60000,
                wait_until="networkidle",
            )
            page.screenshot(path=temp / f"{html.stem}.png", full_page=True)
            browser.close()
    png_files = [str(path) for path in target.glob("*.png")]
    pngs = [Image.open(png) for png in sorted(png_files)]
    imageio.mimsave("test.gif", pngs, fps=2)


if __name__ == "__main__":
    html_path = Path.cwd() / "data" / "visual" / "html"
    ani_path = Path.cwd() / "data" / "visual" / "ani"
